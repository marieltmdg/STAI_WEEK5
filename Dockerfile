FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements-app.txt .
RUN python -m venv /opt/venv
RUN /opt/venv/bin/python -m pip install --upgrade pip
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-compile -r requirements-app.txt
RUN find /opt/venv -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -prune -exec rm -rf {} + \
    && find /opt/venv -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
RUN rm -rf \
    /opt/venv/lib/python3.12/site-packages/pyarrow \
    /opt/venv/lib/python3.12/site-packages/pyarrow-*.dist-info \
    /opt/venv/lib/python3.12/site-packages/pydeck \
    /opt/venv/lib/python3.12/site-packages/pydeck-*.dist-info \
    /opt/venv/lib/python3.12/site-packages/pandas \
    /opt/venv/lib/python3.12/site-packages/pandas-*.dist-info \
    /opt/venv/lib/python3.12/site-packages/numpy \
    /opt/venv/lib/python3.12/site-packages/numpy.libs \
    /opt/venv/lib/python3.12/site-packages/numpy-*.dist-info

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

COPY . .

EXPOSE 8501 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
