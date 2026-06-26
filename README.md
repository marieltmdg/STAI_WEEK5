# Dual-Channel Handbook Support Bot

This project ships the Week 3 RAG system and Week 4 memory/guardrail system through two app channels:

1. Streamlit chat app with message history, streamed responses, and PDF upload.
2. FastAPI endpoint with `/chat` and `/chat/stream`.
3. Structured JSONL LLMOps logging.
4. Dockerfile that starts both servers.

The Week 3 and Week 4 notebook logic has been moved into Python modules. The runtime app does not import or execute `.ipynb` files, so Streamlit, FastAPI, and Docker can start quickly and predictably.

## Implementation Provenance

The production Python files lift the final notebook patterns instead of sourcing notebooks at runtime:

- Week 3 notebook cells 76, 92, 101, and 131 map to [handbook_support_bot.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/handbook_support_bot.py): PDF loading, chunking, local Ollama embeddings, persistent vector storage, MMR retrieval, `gemma3:1b` at `temperature=0.0`, and the strict `Data Not Found` grounded prompt. The deployed retriever uses `k=4` and `fetch_k=16` to improve coverage for short policy questions while preserving the Week 3 MMR approach.
- Week 4 notebook cells 23, 69, and 71 map to [handbook_support_bot.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/handbook_support_bot.py): PII redaction, persistent embedded memory with summary/recent conversation context, semantic recall, input policy guardrails, output guardrails, and safe memory storage after redaction.
- Week 5 delivery requirements are implemented in [chat_delivery_gateway.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/chat_delivery_gateway.py): shared Streamlit/FastAPI gateway, response streaming, SSE events, and JSONL LLMOps logging.

## 1. Prerequisites

Install these before running the app:

- Python 3.12 or 3.13
- Ollama
- Docker Desktop, only if you want to run the container

Start Ollama, then pull the same local models used by the lab notebooks:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

## 2. Create The Environment

From this folder, create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the app dependencies:

```bash
python -m pip install -r requirements-app.txt
```

`requirements.txt` mirrors the runtime dependency set for simple local installs.

## 3. Verify The Files

The required runtime files are:

- [handbook_support_bot.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/handbook_support_bot.py): handbook RAG, memory, and guardrails.
- [chat_delivery_gateway.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/chat_delivery_gateway.py): shared chat, streaming, SSE, and JSONL logging layer.
- [streamlit_app.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/streamlit_app.py): Streamlit chat app.
- [api.py](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/api.py): FastAPI app.
- [Dockerfile](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/Dockerfile): multi-stage container build.
- [school_handbook.pdf](/C:/Users/marie/OneDrive/Documents/School/DLSU25-26/TERM3/STAI100/STAI_WEEK5/school_handbook.pdf): default PDF source.

The first run creates or reuses:

- `handbook_vector_db/`
- `student_memory_db/`
- `llmops_logs/requests.jsonl`

## 4. Run Streamlit

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Use the chat input to ask handbook questions. To test upload support, upload another PDF in the app; that uploaded PDF gets its own temporary vector store for the session.

Uploaded PDFs are saved to a temporary session folder, parsed with `pypdf`, chunked locally, embedded with `nomic-embed-text`, and stored in a temporary JSON vector index for that uploaded document. The uploaded PDF and its vector index are session-local temporary files, not permanent project files.

## 5. Run FastAPI

In a second terminal with the same virtual environment activated:

```bash
python -m uvicorn api:app --reload
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Send a normal chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"What happens if I miss an exam because of a family emergency?\",\"session_id\":\"demo\"}"
```

## 6. Test SSE Streaming

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"What is the standard uniform on a normal Tuesday?\",\"session_id\":\"demo\"}"
```

The response streams Server-Sent Events with `meta`, `chunk`, and `done` events.

## 7. Check LLMOps Logs

After each completed request, check:

```bash
type llmops_logs\requests.jsonl
```

Each JSON line includes:

- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `model`
- `estimated_cost_usd`
- `route`
- `session_id`
- `status`

## 8. Run With Docker

Build the image:

```bash
docker build -t handbook-support-bot .
```

Run both servers from one container:

```bash
docker run --rm -p 8501:8501 -p 8000:8000 ^
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ^
  handbook-support-bot
```

Then open:

```text
http://localhost:8501
```

FastAPI is available at:

```text
http://localhost:8000
```

## 9. Deploy On Render

Render can build this project from the Dockerfile.

1. Push this folder to GitHub.
2. In Render, choose **New > Web Service**.
3. Connect the GitHub repository.
4. Set **Language** to `Docker`.
5. Keep the Dockerfile path as `Dockerfile`.
6. Add an environment variable:

```text
OLLAMA_BASE_URL=<public Ollama-compatible endpoint>
```

Render cannot use `http://localhost:11434` from your laptop. For deployment, Ollama must run on a public or Render-private service that this app can reach.

Render forwards one public HTTP port per web service. This Dockerfile binds Streamlit to Render's `PORT` value, so the public Render URL opens the Streamlit app. The FastAPI server still starts inside the container on port `8000`; deploy it as a second Render service if you need the API public too.

## 10. Troubleshooting

If the bot fails to initialize, confirm Ollama is running and both models are pulled.

If answers look stale after changing `school_handbook.pdf`, delete `handbook_vector_db/` and rerun the app so it can rebuild the vector store.

If Docker cannot reach Ollama on your host machine, confirm the `OLLAMA_BASE_URL` value for your Docker setup. On Docker Desktop, `http://host.docker.internal:11434` usually points back to Ollama running on Windows.
