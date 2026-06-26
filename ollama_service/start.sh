#!/bin/sh
set -eu

export OLLAMA_HOST=0.0.0.0:${PORT:-11434}

ollama serve &
OLLAMA_PID=$!

sleep 5

ollama pull gemma3:1b
ollama pull nomic-embed-text

wait "$OLLAMA_PID"
