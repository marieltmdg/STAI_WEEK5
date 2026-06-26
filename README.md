# Dual-Channel Handbook Support Bot

## Prerequisites

- Python 3.12+
- Ollama
- Docker Desktop, for Docker runs

Start Ollama and pull the required models:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

## Local Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
```

## Run Streamlit

```bash
python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Run FastAPI

```bash
python -m uvicorn api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"What is the standard uniform on a normal Tuesday?\",\"session_id\":\"demo\"}"
```

Streaming request:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"What happens if my GPA is below 2.00?\",\"session_id\":\"demo\"}"
```

## Run With Docker

```bash
docker build -t handbook-support-bot .
docker run --rm -p 8501:8501 -p 8000:8000 ^
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ^
  handbook-support-bot
```

Open:

```text
http://localhost:8501
```

FastAPI:

```text
http://localhost:8000
```

## Logs

```bash
type llmops_logs\requests.jsonl
```

## Public URL

```text
https://stai-week5.onrender.com
```
