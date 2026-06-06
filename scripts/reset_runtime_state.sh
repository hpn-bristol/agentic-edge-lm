#!/usr/bin/env bash
set -euo pipefail

# Reset helper for repeated MA runs. It clears repo-local controller state and
# runs the image pre-pull/reset utility used by the Kubernetes deployment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ELLM_STATE_DIR:-$REPO_ROOT/runtime_state}"
POLICY_DIR="${ELLM_POLICY_DIR:-$STATE_DIR/llm_policy}"

echo "[reset] repo=$REPO_ROOT"
echo "[reset] state=$STATE_DIR"

mkdir -p "$POLICY_DIR"
rm -f "$POLICY_DIR"/macro_policy.json \
      "$POLICY_DIR"/macro_history.jsonl \
      "$POLICY_DIR"/micro_history.jsonl \
      "$POLICY_DIR"/current_policy.json \
      "$POLICY_DIR"/macro_input.json \
      "$POLICY_DIR"/micro_input_last.txt \
      "$POLICY_DIR"/micro_output_last.json \
      "$POLICY_DIR"/agent_events.jsonl

echo "[reset] clearing cluster runtime state"
python3 "$REPO_ROOT/scripts/reset_cluster_runtime.py"

echo "[reset] running image pre-pull/reset utility"
python3 "$REPO_ROOT/image_prepull/preset.py"

echo "[reset] done"
