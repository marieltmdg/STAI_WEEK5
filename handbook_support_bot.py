from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

# Chroma imports OpenTelemetry at module import time. Disabling the SDK keeps
# Streamlit startup/shutdown from tripping telemetry resource detectors locally.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

from chat_delivery_gateway import build_gateway


DEFAULT_HANDBOOK_PATH = Path("school_handbook.pdf")
DEFAULT_MEMORY_PATH = Path("student_memory_db")
DEFAULT_VECTOR_PATH = Path("handbook_vector_db")
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_VECTOR_COLLECTION = "handbook_chunks"


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


class ChromaSummaryBufferMemory:
    def __init__(
        self,
        llm,
        student_id: str,
        memory_path: Path = DEFAULT_MEMORY_PATH,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ):
        self.llm = llm
        self._client = chromadb.PersistentClient(path=str(memory_path))
        self.store = self._client.get_or_create_collection(
            name=f"mem_{student_id.replace('-', '_')}",
            embedding_function=OllamaEmbeddingFunction(
                url=f"{ollama_base_url.rstrip('/')}/api/embeddings",
                model_name=DEFAULT_EMBEDDING_MODEL,
            ),
        )
        self.buffer: list[str] = []
        self.turn_history: list[str] = []
        self.running_summary = ""
        self.buffer_limit = 6
        self.history_limit = 12

    def remember(self, user_input: str, bot_response: str):
        interaction = f"Student: {user_input}\nBot: {bot_response}"
        self.store.add(documents=[interaction], ids=[str(uuid4())])
        self.turn_history.append(interaction)
        self.turn_history = self.turn_history[-self.history_limit:]
        self.buffer.append(interaction)

        if len(self.buffer) >= self.buffer_limit:
            summary_prompt = f"""
Summarize this conversation concisely. Preserve the order of important student questions, preferences, and specific issues.
Prior Summary: {self.running_summary}
Recent Chat: {" | ".join(self.buffer)}
"""
            self.running_summary = self.llm.invoke(summary_prompt).strip()
            self.buffer = []

    def get_context(self, query: str) -> str:
        past_docs = ""
        if self.store.count() > 0:
            docs = self.store.query(query_texts=[query], n_results=min(4, self.store.count()))
            past_docs = "\n".join(docs["documents"][0])

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
        memory_llm=None,
        memory_path: Path = DEFAULT_MEMORY_PATH,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ):
        self.llm = llm_chain
        self.memory = ChromaSummaryBufferMemory(
            memory_llm or llm_chain,
            student_id,
            memory_path=memory_path,
            ollama_base_url=ollama_base_url,
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
    chunks: list
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

    embeddings_model = OllamaEmbeddings(model=embedding_model_name, base_url=ollama_base_url)
    def open_vector_db() -> Chroma:
        return Chroma(
            collection_name=DEFAULT_VECTOR_COLLECTION,
            persist_directory=str(vector_path),
            embedding_function=embeddings_model,
        )

    try:
        db = open_vector_db()
    except Exception as exc:
        if vector_path.exists():
            shutil.rmtree(vector_path)
            db = open_vector_db()
        else:
            raise RuntimeError(f"Could not initialize vector store at {vector_path}: {exc}") from exc

    try:
        existing_count = db._collection.count()
    except Exception as exc:
        if vector_path.exists():
            shutil.rmtree(vector_path)
            db = open_vector_db()
        else:
            raise RuntimeError(f"Could not initialize vector store at {vector_path}: {exc}") from exc
        existing_count = 0

    chunks = []
    if force_rebuild_vector_db or existing_count == 0:
        loader_pdf = PyPDFLoader(str(handbook_path))
        docs_pdf = loader_pdf.load()

        # Week 3 production RAG settings: semantic chunking, local embeddings,
        # Chroma storage, MMR retrieval, and a strict grounded prompt.
        text_splitter = SemanticChunker(
            embeddings_model,
            breakpoint_threshold_type="percentile",
        )
        chunks = text_splitter.split_documents(docs_pdf)
        db.add_documents(chunks)

    good_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    llm_strict = Ollama(model=model_name, temperature=0.0, base_url=ollama_base_url)
    strict_prompt = ChatPromptTemplate.from_template(
        """
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
    )
    strict_chain = strict_prompt | llm_strict | StrOutputParser()

    secure_bot = SecureStudentBot(
        student_id=student_id,
        retriever=good_retriever,
        llm_chain=strict_chain,
        memory_llm=llm_strict,
        memory_path=memory_path,
        ollama_base_url=ollama_base_url,
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
        embeddings_model=embeddings_model,
        db=db,
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
    "DEFAULT_VECTOR_COLLECTION",
    "HandbookBotBundle",
    "SecureStudentBot",
    "ChromaSummaryBufferMemory",
    "build_handbook_bundle",
    "redact_pii",
]
