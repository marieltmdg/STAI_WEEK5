from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate that keeps the logging contract simple and local-only.
    """
    if not text:
        return 0
    words = len(text.split())
    return max(1, math.ceil(words * 1.33))


def _iter_chunks(text: str, chunk_size: int = 24) -> Iterator[str]:
    """
    Split a completed response into small chunks for Streamlit / SSE delivery.
    This keeps the transport streamed even when the underlying bot returns a full string.
    """
    if not text:
        return
    words = text.split()
    if not words:
        yield text
        return

    buffer: list[str] = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= chunk_size:
            yield " ".join(buffer) + " "
            buffer = []
    if buffer:
        yield " ".join(buffer)


def _sse(data: dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def _build_retrieval_query(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized.split()) <= 5 and not normalized.endswith("?"):
        return (
            "Find the handbook section that directly answers this lookup: "
            f"{normalized}. Include exact names, policy headings, requirements, "
            "definitions, and closely related terms."
        )
    return text


@dataclass
class LLMOpsRecord:
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    estimated_cost_usd: float
    request_id: str
    timestamp_utc: str
    route: str
    session_id: str
    status: str
    input_text: str
    output_text: str


class JsonlLLMOpsLogger:
    """
    Minimal structured logger that writes one JSON line per request.
    Optionally mirrors the same payload to MLflow when available.
    """

    def __init__(
        self,
        log_path: str | Path = "logs/llmops.jsonl",
        enable_mlflow: bool = False,
        mlflow_experiment: str = "handbook-support-bot",
    ):
        self.log_path = Path(log_path)
        self.enable_mlflow = enable_mlflow
        self.mlflow_experiment = mlflow_experiment
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: LLMOpsRecord) -> None:
        payload = asdict(record)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        if not self.enable_mlflow:
            return

        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name=record.route):
                mlflow.log_params(
                    {
                        "request_id": record.request_id,
                        "route": record.route,
                        "session_id": record.session_id,
                        "model": record.model,
                    }
                )
                mlflow.log_metrics(
                    {
                        "latency_ms": record.latency_ms,
                        "prompt_tokens": record.prompt_tokens,
                        "completion_tokens": record.completion_tokens,
                        "total_tokens": record.total_tokens,
                        "estimated_cost_usd": record.estimated_cost_usd,
                    }
                )
                mlflow.log_text(json.dumps(payload, indent=2), f"{record.request_id}.json")
        except Exception:
            # Logging must never break the request path.
            pass


class ChatDeliveryGateway:
    """
    Wraps the secure handbook bot and exposes:
    - normal chat calls
    - streaming chat delivery
    - request logging
    - optional Streamlit and FastAPI adapters
    """

    def __init__(
        self,
        bot: Any,
        *,
        model_name: str = "gemma3:1b",
        cost_per_1k_tokens_usd: float = 0.0,
        logger: Optional[JsonlLLMOpsLogger] = None,
    ):
        self.bot = bot
        self.model_name = model_name
        self.cost_per_1k_tokens_usd = cost_per_1k_tokens_usd
        self.logger = logger or JsonlLLMOpsLogger()

    def _build_payload(self, user_input: str) -> dict[str, str]:
        clean_input = user_input
        if hasattr(self.bot, "_redact_pii"):
            clean_input = self.bot._redact_pii(user_input)
        retrieval_query = _build_retrieval_query(clean_input)

        handbook_context = ""
        if hasattr(self.bot, "retriever"):
            docs = self.bot.retriever.invoke(retrieval_query)
            handbook_context = "\n".join(doc.page_content for doc in docs)

        memory_context = ""
        if hasattr(self.bot, "memory") and hasattr(self.bot.memory, "get_context"):
            memory_context = self.bot.memory.get_context(clean_input)

        return {
            "question": clean_input,
            "handbook_context": handbook_context,
            "memory_context": memory_context,
        }

    def _invoke_chain(self, payload: dict[str, str]) -> str:
        if not hasattr(self.bot, "llm_chain"):
            raise AttributeError("bot must expose llm_chain")

        llm = self.bot.llm_chain
        if hasattr(llm, "invoke"):
            return llm.invoke(payload)
        raise AttributeError("bot.llm_chain must implement invoke()")

    def answer(self, user_input: str, *, session_id: str = "default", route: str = "chat") -> str:
        start = time.perf_counter()

        if hasattr(self.bot, "_is_policy_violation") and self.bot._is_policy_violation(user_input):
            final_response = "Request blocked: Your prompt violates university policies."
            latency_ms = int((time.perf_counter() - start) * 1000)
            prompt_tokens = _estimate_tokens(user_input)
            completion_tokens = _estimate_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens
            estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)
            self.logger.write(
                LLMOpsRecord(
                    request_id=f"req_{int(time.time() * 1000)}",
                    timestamp_utc=_utc_now(),
                    route=route,
                    session_id=session_id,
                    model=self.model_name,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                    status="blocked",
                    input_text=user_input,
                    output_text=final_response,
                )
            )
            return final_response

        payload = self._build_payload(user_input)
        try:
            raw_response = self._invoke_chain(payload)
        except Exception as exc:
            final_response = f"LLM Error: {exc}"
            latency_ms = int((time.perf_counter() - start) * 1000)
            prompt_tokens = _estimate_tokens("\n".join([payload["question"], payload["handbook_context"], payload["memory_context"]]))
            completion_tokens = _estimate_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens
            estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)
            self.logger.write(
                LLMOpsRecord(
                    request_id=f"req_{int(time.time() * 1000)}",
                    timestamp_utc=_utc_now(),
                    route=route,
                    session_id=session_id,
                    model=self.model_name,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                    status="error",
                    input_text=user_input,
                    output_text=final_response,
                )
            )
            return final_response

        if hasattr(self.bot, "_redact_pii"):
            clean_response = self.bot._redact_pii(raw_response)
        else:
            clean_response = raw_response

        if hasattr(self.bot, "_output_validator"):
            final_response = self.bot._output_validator(clean_response)
        else:
            final_response = clean_response

        if hasattr(self.bot, "memory") and hasattr(self.bot.memory, "remember"):
            self.bot.memory.remember(user_input=payload["question"], bot_response=final_response)

        latency_ms = int((time.perf_counter() - start) * 1000)
        prompt_text = "\n".join(
            [
                payload["question"],
                payload["handbook_context"],
                payload["memory_context"],
            ]
        )
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(final_response)
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)

        self.logger.write(
            LLMOpsRecord(
                request_id=f"req_{int(time.time() * 1000)}",
                timestamp_utc=_utc_now(),
                route=route,
                session_id=session_id,
                model=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
                status="ok",
                input_text=user_input,
                output_text=final_response,
            )
        )
        return final_response

    def stream_answer(
        self,
        user_input: str,
        *,
        session_id: str = "default",
        route: str = "chat/stream",
        chunk_size: int = 24,
    ) -> Iterator[str]:
        """
        Stream a finalized response in chunks while still preserving the Week 4 safety gates.
        """
        start = time.perf_counter()

        if hasattr(self.bot, "_is_policy_violation") and self.bot._is_policy_violation(user_input):
            final_response = "Request blocked: Your prompt violates university policies."
            for chunk in _iter_chunks(final_response, chunk_size=chunk_size):
                yield chunk
            latency_ms = int((time.perf_counter() - start) * 1000)
            prompt_tokens = _estimate_tokens(user_input)
            completion_tokens = _estimate_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens
            estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)
            self.logger.write(
                LLMOpsRecord(
                    request_id=f"req_{int(time.time() * 1000)}",
                    timestamp_utc=_utc_now(),
                    route=route,
                    session_id=session_id,
                    model=self.model_name,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                    status="blocked",
                    input_text=user_input,
                    output_text=final_response,
                )
            )
            return

        payload = self._build_payload(user_input)
        try:
            raw_response = self._invoke_chain(payload)
        except Exception as exc:
            final_response = f"LLM Error: {exc}"
            for chunk in _iter_chunks(final_response, chunk_size=chunk_size):
                yield chunk
            latency_ms = int((time.perf_counter() - start) * 1000)
            prompt_tokens = _estimate_tokens("\n".join([payload["question"], payload["handbook_context"], payload["memory_context"]]))
            completion_tokens = _estimate_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens
            estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)
            self.logger.write(
                LLMOpsRecord(
                    request_id=f"req_{int(time.time() * 1000)}",
                    timestamp_utc=_utc_now(),
                    route=route,
                    session_id=session_id,
                    model=self.model_name,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                    status="error",
                    input_text=user_input,
                    output_text=final_response,
                )
            )
            return

        if hasattr(self.bot, "_redact_pii"):
            clean_response = self.bot._redact_pii(raw_response)
        else:
            clean_response = raw_response

        if hasattr(self.bot, "_output_validator"):
            final_response = self.bot._output_validator(clean_response)
        else:
            final_response = clean_response

        if hasattr(self.bot, "memory") and hasattr(self.bot.memory, "remember"):
            self.bot.memory.remember(user_input=payload["question"], bot_response=final_response)

        for chunk in _iter_chunks(final_response, chunk_size=chunk_size):
            yield chunk

        latency_ms = int((time.perf_counter() - start) * 1000)
        prompt_text = "\n".join(
            [
                payload["question"],
                payload["handbook_context"],
                payload["memory_context"],
            ]
        )
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(final_response)
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)

        self.logger.write(
            LLMOpsRecord(
                request_id=f"req_{int(time.time() * 1000)}",
                timestamp_utc=_utc_now(),
                route=route,
                session_id=session_id,
                model=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
                status="ok",
                input_text=user_input,
                output_text=final_response,
            )
        )

    def sse_events(
        self,
        user_input: str,
        *,
        session_id: str = "default",
        route: str = "chat/stream",
        chunk_size: int = 24,
    ) -> Iterator[str]:
        yield _sse({"status": "started", "route": route, "session_id": session_id}, event="meta")
        for chunk in self.stream_answer(
            user_input,
            session_id=session_id,
            route=route,
            chunk_size=chunk_size,
        ):
            yield _sse({"chunk": chunk}, event="chunk")
        yield _sse({"status": "completed"}, event="done")


DualChannelGateway = ChatDeliveryGateway


def build_gateway(
    bot: Any,
    *,
    model_name: str = "gemma3:1b",
    cost_per_1k_tokens_usd: float = 0.0,
    log_path: str | Path = "logs/llmops.jsonl",
    enable_mlflow: bool = False,
) -> ChatDeliveryGateway:
    logger = JsonlLLMOpsLogger(
        log_path=log_path,
        enable_mlflow=enable_mlflow,
    )
    return ChatDeliveryGateway(
        bot,
        model_name=model_name,
        cost_per_1k_tokens_usd=cost_per_1k_tokens_usd,
        logger=logger,
    )


def create_fastapi_app(gateway: ChatDeliveryGateway):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        message: str
        session_id: str = "default"

    app = FastAPI(title="Handbook Support Bot", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat")
    def chat(payload: ChatRequest):
        return {
            "message": gateway.answer(payload.message, session_id=payload.session_id, route="chat"),
        }

    @app.post("/chat/stream")
    def chat_stream(payload: ChatRequest):
        return StreamingResponse(
            gateway.sse_events(
                payload.message,
                session_id=payload.session_id,
                route="chat/stream",
            ),
            media_type="text/event-stream",
        )

    return app


def render_streamlit_app(
    gateway: ChatDeliveryGateway,
    *,
    title: str = "Handbook Support Bot",
    subtitle: str = "RAG + memory + guardrails with Streamlit chat and SSE logging",
    on_pdf_upload: Optional[Callable[[Any], None]] = None,
    configure_page: bool = True,
) -> None:
    import streamlit as st

    if configure_page:
        st.set_page_config(page_title=title, page_icon="chat", layout="wide")
        st.title(title)
        st.caption(subtitle)

    current_gateway = st.session_state.get("gateway", gateway)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_pdf = st.file_uploader("Upload a PDF for this session", type=["pdf"])
    if uploaded_pdf is not None:
        upload_key = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
        if on_pdf_upload is not None and st.session_state.get("uploaded_pdf_key") != upload_key:
            updated_gateway = on_pdf_upload(uploaded_pdf)
            if updated_gateway is not None:
                current_gateway = updated_gateway
                st.session_state.gateway = updated_gateway
                st.session_state.uploaded_pdf_key = upload_key
        st.info(f"Loaded {uploaded_pdf.name} for this session.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the bot anything about the handbook")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = []
        for chunk in current_gateway.stream_answer(prompt, route="streamlit"):
            collected.append(chunk)
            placeholder.markdown("".join(collected))

    reply = "".join(collected)
    st.session_state.messages.append({"role": "assistant", "content": reply})


__all__ = [
    "DualChannelGateway",
    "ChatDeliveryGateway",
    "JsonlLLMOpsLogger",
    "LLMOpsRecord",
    "build_gateway",
    "create_fastapi_app",
    "render_streamlit_app",
]
