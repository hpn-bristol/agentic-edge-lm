#!/usr/bin/env bash
set -euo pipefail

# Start one MA controller run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export ELLM_STATE_DIR="${ELLM_STATE_DIR:-$REPO_ROOT/runtime_state}"
export ELLM_POLICY_DIR="${ELLM_POLICY_DIR:-$ELLM_STATE_DIR/llm_policy}"
export LLM_ROUTING_DBNAME="${LLM_ROUTING_DBNAME:-ma-runtime-$(date +%s)}"
export PYTHONUNBUFFERED=1

mkdir -p "$ELLM_POLICY_DIR"

echo "[ma] repo=$REPO_ROOT"
echo "[ma] state=$ELLM_STATE_DIR"
echo "[ma] db=$LLM_ROUTING_DBNAME"
echo "[ma] starting controller"

python3 -u -m MA.ctdema.llm.policy_manager
