#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="Qwen/Qwen2.5-3B-Instruct"
HOST="127.0.0.1"
PORT="30000"
BACKEND="triton"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_relaykv_server.sh off [--model MODEL] [--host HOST] [--port PORT] [--backend BACKEND]
  ./scripts/start_relaykv_server.sh on BLOCKS [--model MODEL] [--host HOST] [--port PORT] [--backend BACKEND]
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

BLOCKS=""
if [[ "$MODE" == "on" ]]; then
  if [[ $# -lt 1 || "$1" == --* ]]; then
    echo "error: on mode requires BLOCKS" >&2
    usage
    exit 1
  fi
  BLOCKS="$1"
  shift
elif [[ "$MODE" != "off" ]]; then
  echo "error: mode must be 'off' or 'on'" >&2
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

if [[ "$MODE" == "off" ]]; then
  unset RELAYKV_V0_APPLY
  unset RELAYKV_V0_RETRIEVAL_BLOCKS
else
  export RELAYKV_V0_APPLY=1
  export RELAYKV_V0_RETRIEVAL_BLOCKS="$BLOCKS"
fi

echo "mode: $MODE"
echo "model: $MODEL"
echo "host: $HOST"
echo "port: $PORT"
echo "backend: $BACKEND"
echo "RELAYKV_V0_APPLY: ${RELAYKV_V0_APPLY-}"
echo "RELAYKV_V0_RETRIEVAL_BLOCKS: ${RELAYKV_V0_RETRIEVAL_BLOCKS-}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend "$BACKEND"
