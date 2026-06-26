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
        upload_root = Path(st.session_state.setdefault("knowledge_base_root", tempfile.mkdtemp(prefix="handbook_upload_")))
        student_id = st.session_state.setdefault("knowledge_base_student_id", f"expanded_{uuid4().hex}")
        uploaded_paths = st.session_state.setdefault("uploaded_pdf_paths", [])
        uploaded_names = st.session_state.setdefault("uploaded_pdf_names", [])

        safe_name = Path(uploaded_file.name).name
        temp_path = upload_root / f"{uuid4().hex}_{safe_name}"
        temp_path.write_bytes(uploaded_file.getbuffer())
        uploaded_paths.append(str(temp_path))
        uploaded_names.append(safe_name)

        rebuilt = build_handbook_bundle(
            additional_handbook_paths=[Path(path) for path in uploaded_paths],
            student_id=student_id,
            memory_path=upload_root / "student_memory_db",
            vector_path=upload_root / "handbook_vector_db",
            force_rebuild_vector_db=True,
        )
        st.session_state.uploaded_pdf_chunk_count = len(rebuilt.chunks)
        st.session_state.knowledge_base_label = "school_handbook.pdf + " + " + ".join(uploaded_names)
        return rebuilt.chat_gateway

    render_streamlit_app(
        bundle.chat_gateway,
        title="Oakridge Academy Chatbot",
        subtitle="Includes RAG, memory, LLMOps logging, and 3-layer guardrails",
        on_pdf_upload=handle_pdf_upload,
        configure_page=False,
    )


if __name__ == "__main__":
    main()
