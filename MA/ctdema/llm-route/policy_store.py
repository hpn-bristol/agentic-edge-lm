"""Persistence helpers for router policies."""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, Iterable, Optional


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_STATE_ROOT = pathlib.Path(os.getenv("ELLM_STATE_DIR", str(_REPO_ROOT / "runtime_state")))
_POLICY_DIR = pathlib.Path(os.getenv("ELLM_POLICY_DIR", str(_STATE_ROOT / "llm_policy")))
DEFAULT_POLICY_FILE = _POLICY_DIR / "current_policy.json"


class PolicyStore:
    """Lightweight file-backed policy repository."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or DEFAULT_POLICY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, policy: Dict[str, Any]) -> None:
        payload = json.dumps(policy, ensure_ascii=False, indent=2)
        self.path.write_text(payload, encoding="utf-8")

    def needs_refresh(self, policy: Optional[Dict[str, Any]]) -> bool:
        """Refresh as soon as the underlying policy file changes.

        We ignore ttl_slots here and simply compare the file's mtime with
        the timestamp recorded when the policy was last loaded.
        """
        if not policy:
            return True
        if not self.path.exists():
            return False
        meta = policy.get("meta", {})
        try:
            last_mtime = float(meta.get("loaded_mtime", 0.0))
        except Exception:
            last_mtime = 0.0
        current_mtime = self.path.stat().st_mtime
        return current_mtime > last_mtime

    def stamp_loaded(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        policy = dict(policy)
        meta = dict(policy.get("meta", {}))
        # Record both the wall-clock time and the file mtime at which this
        # policy snapshot was observed by the router loop.
        meta["loaded_at_ts"] = time.time()
        try:
            meta["loaded_mtime"] = self.path.stat().st_mtime
        except FileNotFoundError:
            meta["loaded_mtime"] = meta.get("loaded_at_ts", time.time())
        policy["meta"] = meta
        return policy

    # ------------------------------------------------------------------
    @staticmethod
    def build_uniform_policy(nodes: Iterable[str], anchors: Iterable[str] = ("1", "2", "3", "4")) -> Dict[str, Any]:
        nodes = list(nodes)
        anchors = list(anchors)
        if not nodes:
            return {"routes": {anchor: {} for anchor in anchors}, "meta": {}}
        uniform = 1.0 / len(nodes)
        routes = {anchor: {node: uniform for node in nodes} for anchor in anchors}
        return {
            "routes": routes,
            "meta": {
                "generated_at": None,
                "ttl_slots": 30,
                "delta_limit": 0.1,
                "ramp_slots": 5,
                "source": "uniform-bootstrap",
            },
        }
