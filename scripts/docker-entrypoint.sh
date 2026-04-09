#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

PORT="${VGGT_SERVE_PORT:-8000}"
HOST="${VGGT_SERVE_HOST:-0.0.0.0}"

exec uvicorn vggt_serve.app:app --host "${HOST}" --port "${PORT}"
