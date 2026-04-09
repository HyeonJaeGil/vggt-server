#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${SERVICE_PORT:-8000}"
GPU_ENABLED=1
DETACH=0
BUILD_ON_UP=1
ACTION="up"
DOCKER_GPUS_VALUE="${DOCKER_GPUS:-all}"
SHM_SIZE_VALUE="${SHM_SIZE:-8gb}"

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_compose.sh [action] [options]

Actions:
  up         Build and start the service. Default action.
  down       Stop the service and remove containers.
  build      Build the service image only.
  logs       Follow service logs.
  ps         Show compose service status.

Options:
  -p, --port PORT     Forward this host/container port. Default: 8000
  --cpu               Use the base compose file only and skip GPU settings.
  --gpus VALUE        GPU selector passed to compose. Default: all
  --shm-size VALUE    Shared memory size for GPU compose. Default: 8gb
  -d, --detach        Run 'up' in detached mode.
  --no-build          Skip '--build' on 'up'.
  -h, --help          Show this help text.

Examples:
  scripts/docker_compose.sh up --port 9000
  scripts/docker_compose.sh up --port 9000 --cpu
  scripts/docker_compose.sh logs --port 9000
  scripts/docker_compose.sh down
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    up|down|build|logs|ps)
      ACTION="$1"
      shift
      ;;
    -p|--port)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    --cpu)
      GPU_ENABLED=0
      shift
      ;;
    --gpus)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      DOCKER_GPUS_VALUE="$2"
      shift 2
      ;;
    --shm-size)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      SHM_SIZE_VALUE="$2"
      shift 2
      ;;
    -d|--detach)
      DETACH=1
      shift
      ;;
    --no-build)
      BUILD_ON_UP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "Port must be an integer between 1 and 65535: $PORT" >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/.gitmodules" ] || [ ! -f "${ROOT_DIR}/vggt/pyproject.toml" ]; then
  echo "VGGT submodule is missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

COMPOSE_ARGS=(-f docker-compose.yml)
if [ "$GPU_ENABLED" -eq 1 ]; then
  COMPOSE_ARGS+=(-f docker-compose.gpu.yml)
fi

export SERVICE_PORT="$PORT"
export DOCKER_GPUS="$DOCKER_GPUS_VALUE"
export SHM_SIZE="$SHM_SIZE_VALUE"

cd "$ROOT_DIR"

case "$ACTION" in
  up)
    UP_ARGS=(up)
    if [ "$BUILD_ON_UP" -eq 1 ]; then
      UP_ARGS+=(--build)
    fi
    if [ "$DETACH" -eq 1 ]; then
      UP_ARGS+=(-d)
    fi
    exec docker compose "${COMPOSE_ARGS[@]}" "${UP_ARGS[@]}"
    ;;
  down)
    exec docker compose "${COMPOSE_ARGS[@]}" down
    ;;
  build)
    exec docker compose "${COMPOSE_ARGS[@]}" build
    ;;
  logs)
    exec docker compose "${COMPOSE_ARGS[@]}" logs -f vggt-serve
    ;;
  ps)
    exec docker compose "${COMPOSE_ARGS[@]}" ps
    ;;
esac
