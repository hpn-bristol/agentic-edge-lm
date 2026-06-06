"""Best-effort structured logger for LLM-agent invocations.

Each time one of our three LLM agent layers (Tier-1 macro planner,
Tier-2 micro router, Tier-2 on-node deployment controller) calls an
LLM, we record the outcome of that call so that we can later quantify
how often the LLM produced a usable answer vs. how often the system
used a deterministic fallback path.

Records are written to MongoDB collection ``agent_events.<dbname>``
where ``<dbname>`` is the run-scoped database name passed via env
``LLM_ROUTING_DBNAME``. If MongoDB is unavailable, records are
appended to a JSONL file as a fallback so the runtime is never
blocked by logging.

Schema of one record::

  {
    "ts":            <unix epoch seconds, float>,
    "layer":         "macro" | "micro" | "local",
    "outcome":       "success" | "parse_fail" | "validation_fail"
                    | "api_error" | "fallback_random"
                    | "fallback_keep_local",
    "epoch_id":      <int|None>,
    "slot_index":    <int|None>,
    "node":          <str|None>,            # for local layer
    "model":         <str|None>,
    "tokens_in":     <int|None>,
    "tokens_out":    <int|None>,
    "latency_ms":    <int|None>,
    "validation_errors": [<str>, ...],      # only when validation_fail
    "raw_truncated": <str max 400>,         # only when parse/validation fail
    "error":         <str max 300>,         # only when api_error
  }

This module is intentionally dependency-light. The MongoDB write is
attempted lazily and never raises into the caller.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dep
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore


VALID_OUTCOMES = {
    "success",
    "parse_fail",
    "validation_fail",
    "api_error",
    "fallback_random",
    "fallback_keep_local",
}


def _truncate(s: Any, n: int) -> Optional[str]:
    if s is None:
        return None
    try:
        s = str(s)
    except Exception:
        return None
    return s if len(s) <= n else s[:n] + "...[truncated]"


class AgentEventLogger:
    """Append agent events to MongoDB; fall back to a local JSONL file."""

    def __init__(
        self,
        layer: str,
        dbname: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        fallback_path: Optional[str] = None,
    ) -> None:
        if layer not in ("macro", "micro", "local"):
            raise ValueError(f"Unknown layer: {layer!r}")
        self.layer = layer

        self._dbname = dbname or os.getenv("LLM_ROUTING_DBNAME") or ""
        self._mongo_uri = mongo_uri or os.getenv("LLM_MONGO_URI") or ""
        self._client: Optional[Any] = None
        self._coll: Optional[Any] = None
        self._mongo_ready: Optional[bool] = None  # tri-state: None=not tried

        # Container-friendly default; outside the container the caller can
        # still pass a custom fallback_path.
        default_fb = Path("/app/results/agent_events.jsonl")
        self._fallback_path = Path(fallback_path) if fallback_path else default_fb

    # ------------------------------------------------------------------
    def set_dbname(self, dbname: str) -> None:
        """Update the dbname after construction (e.g. on /set_dbname)."""
        if dbname and dbname != self._dbname:
            self._dbname = dbname
            # Force re-resolve on next log()
            self._client = None
            self._coll = None
            self._mongo_ready = None

    # ------------------------------------------------------------------
    def _ensure_mongo(self) -> bool:
        if self._mongo_ready is not None:
            return self._mongo_ready
        if MongoClient is None or not self._dbname:
            self._mongo_ready = False
            return False
        try:
            uri = self._mongo_uri or self._resolve_mongo_uri()
            if not uri:
                self._mongo_ready = False
                return False
            self._client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            self._coll = self._client["agent_events"][self._dbname]
            self._mongo_ready = True
            return True
        except Exception:
            self._mongo_ready = False
            return False

    @staticmethod
    def _resolve_mongo_uri() -> str:
        """Resolve mongodb-service IP via the in-cluster Kubernetes API.

        Returns an empty string if discovery fails. Cached at the
        instance level via _ensure_mongo so this only runs once.
        """
        try:
            from kubernetes import client as k8s_client, config as k8s_config  # type: ignore
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            v1 = k8s_client.CoreV1Api()
            svc = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
            return f"mongodb://root:secret@{svc.spec.cluster_ip}:27017"
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def log(
        self,
        outcome: str,
        *,
        epoch_id: Optional[int] = None,
        slot_index: Optional[int] = None,
        node: Optional[str] = None,
        model: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        latency_ms: Optional[int] = None,
        validation_errors: Optional[List[str]] = None,
        raw_truncated: Optional[str] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Best-effort write of a single event. Never raises."""
        try:
            if outcome not in VALID_OUTCOMES:
                outcome = f"unknown:{outcome}"

            record: Dict[str, Any] = {
                "ts": time.time(),
                "layer": self.layer,
                "outcome": outcome,
            }
            if epoch_id is not None:
                record["epoch_id"] = int(epoch_id)
            if slot_index is not None:
                record["slot_index"] = int(slot_index)
            if node:
                record["node"] = str(node)
            if model:
                record["model"] = str(model)
            if tokens_in is not None:
                record["tokens_in"] = int(tokens_in)
            if tokens_out is not None:
                record["tokens_out"] = int(tokens_out)
            if latency_ms is not None:
                record["latency_ms"] = int(latency_ms)
            if validation_errors:
                record["validation_errors"] = [str(x)[:200] for x in validation_errors][:10]
            if raw_truncated is not None:
                record["raw_truncated"] = _truncate(raw_truncated, 400)
            if error is not None:
                record["error"] = _truncate(error, 300)
            for k, v in extra.items():
                if v is None:
                    continue
                record[k] = v

            wrote = False
            if self._ensure_mongo():
                try:
                    self._coll.insert_one(record)  # type: ignore[union-attr]
                    wrote = True
                except Exception:
                    wrote = False
                    # invalidate cached connection so the next call may retry
                    self._mongo_ready = None
            if not wrote:
                self._append_fallback(record)
        except Exception:
            # Logging must never crash the caller.
            pass

    # ------------------------------------------------------------------
    def _append_fallback(self, record: Dict[str, Any]) -> None:
        try:
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._fallback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


# Convenience helper to extract OpenAI usage tokens robustly.
def get_openai_usage(completion: Any) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {"tokens_in": None, "tokens_out": None}
    try:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return out
        ti = getattr(usage, "prompt_tokens", None)
        to = getattr(usage, "completion_tokens", None)
        out["tokens_in"] = int(ti) if ti is not None else None
        out["tokens_out"] = int(to) if to is not None else None
    except Exception:
        pass
    return out
