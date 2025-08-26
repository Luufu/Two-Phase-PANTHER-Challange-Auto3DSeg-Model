# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user + working dirs
ARG USER=algorithm
ARG UID=10001
RUN useradd -m -u ${UID} -s /bin/bash ${USER} \
 && mkdir -p /opt/algorithm /input /output \
 && chown -R ${USER}:${USER} /opt/algorithm /input /output

USER ${USER}
WORKDIR /opt/algorithm
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# 1) Python deps (pin MONAI to your training version)
COPY --chown=${USER}:${USER} requirements.txt ./requirements.txt
RUN python -m pip install --user -r requirements.txt

# 2) Inference code
COPY --chown=${USER}:${USER} inference ./inference

# 3) Stage-1 and Stage-2 fold artifacts (5 folds each)
# Expect layout like results/segresnet_0/{model/model.pt, configs/hyper_parameters.yaml, scripts/segmenter.py}
COPY --chown=${USER}:${USER} results ./results
COPY --chown=${USER}:${USER} results_stage2 ./results_stage2

# Entry point reads from /input and writes to /output
ENTRYPOINT ["python", "-m", "inference.gc_entrypoint"]
