# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/grahamdwall/phi2-server"

ENV PYTHONUNBUFFERED=1d
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV HF_HOME=/workspace/data/models
ENV TRANSFORMERS_VERBOSITY=debug

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only version
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install core libraries first
# Assuming that RunPod has GPU drivers installed. pip install torch was bloating the image by 9GB
COPY requirements.txt .

# Install torch with CUDA 12.1 from the custom PyTorch index
RUN pip install --no-cache-dir torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Then install the rest of the requirements from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY phi2_api.py .

# Copy your baked-in LLM models
#COPY phi2_model_full/ ./models/microsoft/phi-2

EXPOSE 8000

# for testing
#CMD ["uvicorn", "phi2_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
# for production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "phi2_api:app", "--bind", "0.0.0.0:8080"]
