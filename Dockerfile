FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/cache/huggingface
ENV HF_HUB_CACHE=/workspace/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git git-lfs ffmpeg sox \
    libsndfile1 libsndfile1-dev build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel packaging ninja

RUN python3 -m pip install \
    torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
