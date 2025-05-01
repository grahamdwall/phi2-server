# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/grahamdwall/phi2-server"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV TRANSFORMERS_CACHE=/app/models
ENV TRANSFORMERS_VERBOSITY=debug

WORKDIR /app

# Install core libraries first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY phi2_api.py .

# Copy your baked-in LLM models
COPY phi2_model_full/ ./models/microsoft/phi-2

EXPOSE 8000

CMD ["uvicorn", "phi2_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
