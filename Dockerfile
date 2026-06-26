FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

COPY requirements-app.txt .
RUN python -m pip wheel --wheel-dir /wheels -r requirements-app.txt

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*

COPY . .

EXPOSE 8501 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
