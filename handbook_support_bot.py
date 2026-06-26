from __future__ import annotations

import json
import math
import os
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from chat_delivery_gateway import build_gateway


DEFAULT_HANDBOOK_PATH = Path("school_handbook.pdf")
DEFAULT_MEMORY_PATH = Path("student_memory_db")
DEFAULT_VECTOR_PATH = Path("handbook_vector_db")
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
}


def redact_pii(text: str) -> str:
    text = re.sub(r"\b(?:\+63[-\s]?|0)9\d{2}[-\.\s]?\d{3,4}[-\.\s]?\d{4}\b", "[REDACTED PHONE]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED EMAIL]", text)
    text = re.sub(r"\b\d{1,3}[ -]years?[ -]old\b", "[REDACTED AGE]", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b\d+\s+[A-Za-z][A-Za-z ]+?(?:St(?:reet)?|Ave(?:nue)?|Blvd|Road|Rd|Drive|Dr|Lane|Ln)\.?\b",
        "[REDACTED ADDRESS]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{4}-\d{7}-\d\b", "[REDACTED ID]", text)
    text = re.sub(
        r"(My name is|my name is|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"\1 [REDACTED NAME]",
        text,
    )
    return text


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict[str, object]
    embedding: list[float] | None = None


class OllamaClient:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/api"):
            self.base_url = self.base_url[: -len("/api")]
        self.temperature = temperature

    def _post(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        response = self._post(
            "/api/embeddings",
            {"model": self.embedding_model_name, "prompt": text},
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response did not include an embedding list.")
        return [float(value) for value in embedding]

    def generate(self, prompt: str) -> str:
        response = self._post(
            "/api/generate",
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )
        return str(response.get("response", "")).strip()

    def invoke(self, prompt: str) -> str:
        return self.generate(prompt)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _keyword_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = set(_tokenize(text))
    overlap = sum(1 for token in query_tokens if token in text_tokens)
    score = overlap / len(set(query_tokens))

    normalized_query = " ".join(query.lower().split())
    normalized_text = " ".join(text.lower().split())
    if normalized_query and normalized_query in normalized_text:
        score += 0.5

    return min(score, 1.0)


def _chunk_text(text: str, *, max_words: int = 180, overlap_words: int = 35) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _load_pdf_chunks(handbook_path: Path) -> list[RetrievedDocument]:
    reader = PdfReader(str(handbook_path))
    documents: list[RetrievedDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_number, chunk in enumerate(_chunk_text(page_text), start=1):
            documents.append(
                RetrievedDocument(
                    page_content=chunk,
                    metadata={
                        "source": str(handbook_path),
                        "page": page_number,
                        "chunk": chunk_number,
                    },
                )
            )
    return documents


class JsonVectorStore:
    def __init__(
        self,
        path: Path,
        embedding_client: OllamaClient,
        *,
        source_path: Path,
        embedding_model_name: str,
    ):
        self.path = path
        self.index_path = path / "index.json"
        self.embedding_client = embedding_client
        self.source_path = source_path
        self.embedding_model_name = embedding_model_name
        self.documents: list[RetrievedDocument] = []

    def _source_fingerprint(self) -> dict[str, object]:
        stat = self.source_path.stat()
        return {
            "source": str(self.source_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "embedding_model": self.embedding_model_name,
        }

    def load(self) -> bool:
        if not self.index_path.exists():
            return False

        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            return False

        if payload.get("source_fingerprint") != self._source_fingerprint():
            return False

        self.documents = [
            RetrievedDocument(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
                embedding=[float(value) for value in item.get("embedding", [])],
            )
            for item in payload.get("documents", [])
        ]
        return bool(self.documents)

    def build(self, documents: list[RetrievedDocument]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.documents = []
        for document in documents:
            self.documents.append(
                RetrievedDocument(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    embedding=self.embedding_client.embed(document.page_content),
                )
            )

        payload = {
            "source_fingerprint": self._source_fingerprint(),
            "documents": [
                {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                    "embedding": document.embedding,
                }
                for document in self.documents
            ]
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


class MMRRetriever:
    def __init__(self, store: JsonVectorStore, embedding_client: OllamaClient, *, k: int = 4, fetch_k: int = 16):
        self.store = store
        self.embedding_client = embedding_client
        self.k = k
        self.fetch_k = fetch_k

    def invoke(self, query: str) -> list[RetrievedDocument]:
        query_embedding = self.embedding_client.embed(query)
        scored = [
            (
                document,
                (0.7 * _cosine_similarity(query_embedding, document.embedding or []))
                + (0.3 * _keyword_score(query, document.page_content)),
            )
            for document in self.store.documents
        ]
        candidates = sorted(scored, key=lambda item: item[1], reverse=True)[: self.fetch_k]

        selected: list[RetrievedDocument] = []
        while candidates and len(selected) < self.k:
            if not selected:
                selected.append(candidates.pop(0)[0])
                continue

            best_index = 0
            best_score = float("-inf")
            for index, (candidate, relevance) in enumerate(candidates):
                diversity_penalty = max(
                    _cosine_similarity(candidate.embedding or [], chosen.embedding or [])
                    for chosen in selected
                )
                mmr_score = (0.75 * relevance) - (0.25 * diversity_penalty)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            selected.append(candidates.pop(best_index)[0])

        return selected


class StrictPromptChain:
    def __init__(self, llm: OllamaClient, template: str):
        self.llm = llm
        self.template = template

    def invoke(self, payload: dict[str, str]) -> str:
        prompt = self.template.replace("{handbook_context}", payload["handbook_context"])
        prompt = prompt.replace("{memory_context}", payload["memory_context"])
        prompt = prompt.replace("{question}", payload["question"])
        return self.llm.generate(prompt)


class JsonSummaryBufferMemory:
    def __init__(
        self,
        llm: OllamaClient,
        student_id: str,
        memory_path: Path = DEFAULT_MEMORY_PATH,
    ):
        self.llm = llm
        self.memory_path = memory_path
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_path / f"mem_{student_id.replace('-', '_')}.json"
        self.buffer: list[str] = []
        self.turn_history: list[str] = []
        self.running_summary = ""
        self.buffer_limit = 6
        self.history_limit = 12
        self.records: list[dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return
        self.running_summary = str(payload.get("running_summary", ""))
        self.turn_history = [str(item) for item in payload.get("turn_history", [])]
        self.records = list(payload.get("records", []))

    def _save(self) -> None:
        payload = {
            "running_summary": self.running_summary,
            "turn_history": self.turn_history,
            "records": self.records,
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def remember(self, user_input: str, bot_response: str):
        interaction = f"Student: {user_input}\nBot: {bot_response}"
        try:
            embedding = self.llm.embed(interaction)
        except RuntimeError:
            embedding = []
        self.records.append({"id": str(uuid4()), "text": interaction, "embedding": embedding})
        self.turn_history.append(interaction)
        self.turn_history = self.turn_history[-self.history_limit :]
        self.buffer.append(interaction)

        if len(self.buffer) >= self.buffer_limit:
            summary_prompt = f"""
Summarize this conversation concisely. Preserve the order of important student questions, preferences, and specific issues.
Prior Summary: {self.running_summary}
Recent Chat: {" | ".join(self.buffer)}
"""
            self.running_summary = self.llm.invoke(summary_prompt).strip()
            self.buffer = []

        self._save()

    def get_context(self, query: str) -> str:
        past_docs = ""
        if self.records:
            try:
                query_embedding = self.llm.embed(query)
            except RuntimeError:
                query_embedding = []
            scored = sorted(
                self.records,
                key=lambda record: (
                    _cosine_similarity(query_embedding, [float(value) for value in record.get("embedding", [])])
                    + _keyword_score(query, str(record.get("text", "")))
                ),
                reverse=True,
            )
            past_docs = "\n".join(str(record.get("text", "")) for record in scored[:4])

        return (
            f"[RUNNING SUMMARY]: {self.running_summary if self.running_summary else 'None'}\n\n"
            f"[SEMANTIC RECALL]:\n{past_docs}\n\n"
            f"[RECENT CONVERSATION, OLDEST TO NEWEST]:\n" + "\n".join(self.turn_history)
        )


class SecureStudentBot:
    def __init__(
        self,
        student_id: str,
        retriever,
        llm_chain,
        *,
        memory_llm: OllamaClient,
        memory_path: Path = DEFAULT_MEMORY_PATH,
    ):
        self.llm = llm_chain
        self.memory = JsonSummaryBufferMemory(
            memory_llm,
            student_id,
            memory_path=memory_path,
        )
        self.retriever = retriever
        self.llm_chain = llm_chain

    def _is_policy_violation(self, text: str) -> bool:
        prohibited_keywords = [
            "ignore previous instructions",
            "hacker",
            "poem about space",
            "bypass",
            "system prompt",
            "diagnose",
            "do i have",
            "prescribe",
            "treatment for",
            "cure",
            "what illness",
        ]
        return any(keyword in text.lower() for keyword in prohibited_keywords)

    def _redact_pii(self, text: str) -> str:
        return redact_pii(text)

    def _output_validator(self, response: str) -> str:
        restricted_phrases = ["i am diagnosing you", "my diagnosis is", "you have", "i prescribe"]
        if any(phrase in response.lower() for phrase in restricted_phrases):
            return "[BLOCKED BY OUTPUT GUARDRAIL]: I am an AI assistant, not a doctor."
        return response

    def chat(self, user_input: str, verbose: bool = False) -> str:
        if verbose:
            print(f"[PIPELINE LOG] Original Input: {user_input}")

        if self._is_policy_violation(user_input):
            if verbose:
                print("[PIPELINE LOG] Intercepted by Layer A: Policy Violation Detected.")
            return "Request blocked: Your prompt violates university policies."

        clean_input = self._redact_pii(user_input)

        try:
            handbook_docs = self.retriever.invoke(clean_input)
            handbook_context = "\n".join([doc.page_content for doc in handbook_docs])
            memory_context = self.memory.get_context(clean_input)

            raw_response = self.llm_chain.invoke(
                {
                    "handbook_context": handbook_context,
                    "memory_context": memory_context,
                    "question": clean_input,
                }
            )
        except Exception as exc:
            return f"LLM Error: {exc}"

        clean_response = self._redact_pii(raw_response)
        final_response = self._output_validator(clean_response)

        if verbose and final_response != clean_response:
            print("[PIPELINE LOG] Intercepted by Layer C: Restricted Phrase Blocked.")
        if verbose and clean_input != user_input:
            print(f"[PIPELINE LOG] Intercepted by Layer B: PII Scrubbed -> {clean_input}")

        self.memory.remember(user_input=clean_input, bot_response=final_response)
        return final_response


@dataclass
class HandbookBotBundle:
    docs_path: Path
    chunks: list[RetrievedDocument]
    embeddings_model: object
    db: object
    good_retriever: object
    llm_strict: object
    strict_prompt: object
    strict_chain: object
    secure_bot: SecureStudentBot
    chat_gateway: object


def build_handbook_bundle(
    handbook_path: str | Path = DEFAULT_HANDBOOK_PATH,
    *,
    student_id: str = "student_demo_102",
    model_name: str = DEFAULT_MODEL,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    vector_path: str | Path = DEFAULT_VECTOR_PATH,
    cost_per_1k_tokens_usd: float = 0.0,
    force_rebuild_vector_db: bool = False,
):
    handbook_path = Path(handbook_path)
    memory_path = Path(memory_path)
    vector_path = Path(vector_path)

    if force_rebuild_vector_db and vector_path.exists():
        shutil.rmtree(vector_path)

    llm_strict = OllamaClient(
        model_name=model_name,
        embedding_model_name=embedding_model_name,
        base_url=ollama_base_url,
        temperature=0.0,
    )

    vector_store = JsonVectorStore(
        vector_path,
        llm_strict,
        source_path=handbook_path,
        embedding_model_name=embedding_model_name,
    )
    if force_rebuild_vector_db or not vector_store.load():
        chunks = _load_pdf_chunks(handbook_path)
        if not chunks:
            raise RuntimeError(f"No extractable text was found in {handbook_path}.")
        vector_store.build(chunks)
    else:
        chunks = vector_store.documents

    good_retriever = MMRRetriever(vector_store, llm_strict, k=4, fetch_k=16)
    strict_prompt = """
You are a strict academic assistant. Answer the user's question using the provided context.
Use direct facts from either the retrieved handbook context or the student memory.
If the user asks about the current conversation, answer from STUDENT MEMORY.
For questions about order, such as first or previous questions, use RECENT CONVERSATION in oldest-to-newest order.
If neither the handbook context nor student memory explicitly contains the answer, reply with 'Data Not Found'.

---
RETRIEVED CONTEXT:
{handbook_context}

---
STUDENT MEMORY:
{memory_context}

---
USER QUESTION:
{question}

ANSWER:
"""
    strict_chain = StrictPromptChain(llm_strict, strict_prompt)

    secure_bot = SecureStudentBot(
        student_id=student_id,
        retriever=good_retriever,
        llm_chain=strict_chain,
        memory_llm=llm_strict,
        memory_path=memory_path,
    )

    chat_gateway = build_gateway(
        secure_bot,
        model_name=model_name,
        cost_per_1k_tokens_usd=cost_per_1k_tokens_usd,
        log_path="llmops_logs/requests.jsonl",
        enable_mlflow=False,
    )

    return HandbookBotBundle(
        docs_path=handbook_path,
        chunks=chunks,
        embeddings_model=llm_strict,
        db=vector_store,
        good_retriever=good_retriever,
        llm_strict=llm_strict,
        strict_prompt=strict_prompt,
        strict_chain=strict_chain,
        secure_bot=secure_bot,
        chat_gateway=chat_gateway,
    )


__all__ = [
    "DEFAULT_HANDBOOK_PATH",
    "DEFAULT_MEMORY_PATH",
    "DEFAULT_VECTOR_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "HandbookBotBundle",
    "SecureStudentBot",
    "JsonSummaryBufferMemory",
    "build_handbook_bundle",
    "redact_pii",
]
