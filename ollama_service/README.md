# Ollama Service

Deploy this folder as a separate Docker service.

## Local Test

```bash
docker build -t handbook-ollama ./ollama_service
docker run --rm -p 11434:11434 handbook-ollama
```

Test:

```text
http://localhost:11434/api/tags
```

## Render

Create a separate Render Web Service and use:

```text
Dockerfile Path: ollama_service/Dockerfile
```

After deployment, test:

```text
https://your-ollama-service.onrender.com/api/tags
```

Then set the main app environment variable:

```text
OLLAMA_BASE_URL=https://your-ollama-service.onrender.com
```
