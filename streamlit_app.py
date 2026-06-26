import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import streamlit as st

from chat_delivery_gateway import render_streamlit_app
from handbook_support_bot import build_handbook_bundle


@lru_cache(maxsize=1)
def get_bundle():
    return build_handbook_bundle()


def main():
    st.set_page_config(page_title="Oakridge Academy Chatbot", page_icon="chat", layout="wide")
    st.title("Oakridge Academy Chatbot")
    st.caption("Includes RAG, memory, LLMOps logging, and 3-layer guardrails | Made by Mariel Tamondong")

    splash = st.empty()
    splash.info("Loading the handbook, embeddings, and RAG pipeline...")

    try:
        bundle = get_bundle()
    except Exception as exc:
        splash.error(f"Failed to initialize the bot: {exc}")
        st.stop()

    splash.empty()

    def handle_pdf_upload(uploaded_file):
        upload_root = Path(tempfile.mkdtemp(prefix="handbook_upload_"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=upload_root) as handle:
            handle.write(uploaded_file.getbuffer())
            temp_path = Path(handle.name)

        rebuilt = build_handbook_bundle(
            handbook_path=temp_path,
            student_id=f"uploaded_{uuid4().hex}",
            memory_path=upload_root / "student_memory_db",
            vector_path=upload_root / "handbook_vector_db",
        )
        st.session_state.uploaded_pdf_chunk_count = len(rebuilt.chunks)
        return rebuilt.chat_gateway

    render_streamlit_app(
        bundle.chat_gateway,
        title="Homework: Week 5",
        subtitle="Ship the Week 3-4 system as two channels with full LLMOps logging",
        on_pdf_upload=handle_pdf_upload,
        configure_page=False,
    )


if __name__ == "__main__":
    main()
