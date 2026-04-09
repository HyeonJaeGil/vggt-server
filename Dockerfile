FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniforge.sh \
    && conda clean -afy

WORKDIR /app

COPY environment.yml pyproject.toml README.md ./
COPY scripts ./scripts
COPY vggt ./vggt
COPY vggt_serve ./vggt_serve

RUN conda env create -f environment.yml \
    && conda clean -afy

ENV CONDA_DEFAULT_ENV=vggt-serve-py312 \
    PATH=/opt/conda/envs/vggt-serve-py312/bin:$PATH \
    VGGT_SERVE_HOST=0.0.0.0 \
    VGGT_SERVE_PORT=8000 \
    VGGT_SERVE_DATA_ROOT=/app/data/runs

RUN chmod +x /app/scripts/docker-entrypoint.sh \
    && mkdir -p /app/data/runs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${VGGT_SERVE_PORT:-8000}/healthz" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/docker-entrypoint.sh"]
