FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/cache/huggingface
ENV HF_HUB_CACHE=/workspace/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git git-lfs ffmpeg sox \
    libsndfile1 libsndfile1-dev build-essential cmake curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel packaging ninja

RUN python3 -m pip install \
    torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY constraints.txt /workspace/constraints.txt

# Main env: RunPod + SenseVoice + OmniVoice dependency stack.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    -c /workspace/constraints.txt \
    -r /workspace/requirements.txt

# OmniVoice installed without dependency resolution because deps are pinned above.
RUN python3 -m pip install --no-cache-dir --no-deps omnivoice==0.1.4

# Separate ASR env because OmniASR/fairseq2 requires older huggingface_hub.
RUN python3 -m venv --system-site-packages /opt/earenv && \
    /opt/earenv/bin/python -m pip install -U pip setuptools wheel packaging && \
    /opt/earenv/bin/python -m pip install --no-cache-dir --prefer-binary \
      "huggingface_hub>=0.32,<0.33" \
      "fairseq2[arrow]==0.6" \
      fairseq2n==0.6 \
      pyarrow==24.0.0 \
      pandas==2.3.3 \
      polars==1.40.1 \
      kenlm==0.3.0 \
      retrying \
      xxhash && \
    /opt/earenv/bin/python -m pip install --no-cache-dir --no-deps omnilingual-asr==0.1.0

COPY handler.py /workspace/handler.py
COPY ear_asr_worker.py /workspace/ear_asr_worker.py

# Build-time import smoke test. No model download here.
RUN python3 - <<'PYSMOKE'
import runpod
import torch
from funasr import AutoModel
from omnivoice import OmniVoice
print("MAIN_ENV_IMPORT_OK")
PYSMOKE

RUN /opt/earenv/bin/python /workspace/ear_asr_worker.py --import-only

CMD ["python3", "-u", "/workspace/handler.py"]
