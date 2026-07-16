FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG SUPIR_COMMIT=bda91af2000042f8bedfec8897d92917e67c1d88
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-dev build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
       torch==2.1.2 torchvision==0.16.2 \
    && git clone https://github.com/Fanghua-Yu/SUPIR.git /opt/SUPIR \
    && cd /opt/SUPIR && git checkout ${SUPIR_COMMIT} \
    && python3 -m pip install -r /opt/SUPIR/requirements.txt \
    && python3 -m pip install "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "pydantic-settings>=2.4.0" \
    && sed -i -E "s|^(LLAVA_CLIP_PATH|LLAVA_MODEL_PATH|SDXL_CLIP1_PATH|SDXL_CLIP2_CKPT_PTH) =.*|\1 = None|" /opt/SUPIR/CKPT_PTH.py

WORKDIR /service
COPY supir_worker /service/supir_worker
EXPOSE 8010
CMD ["python3", "-m", "uvicorn", "supir_worker.api:app", "--host", "0.0.0.0", "--port", "8010"]