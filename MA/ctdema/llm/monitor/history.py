"""Simple JSONL history store for snapshots and policies."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict


class HistoryStore:
    """Append-only history writer used by the policy manager."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.base_path / "snapshot_history.jsonl"
        self.policy_path = self.base_path / "policy_history.jsonl"

    @staticmethod
    def _append(path: pathlib.Path, payload: Dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # History is best-effort; swallow errors to avoid disturbing the control loop
            pass

    def append_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._append(self.snapshot_path, snapshot)

    def append_policy(self, payload: Dict[str, Any]) -> None:
        self._append(self.policy_path, payload)

    def iter_snapshots(self, limit: int = 1000) -> list[Dict[str, Any]]:
        if not self.snapshot_path.exists():
            return []
        items: list[Dict[str, Any]] = []
        try:
            with self.snapshot_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        if limit <= 0 or len(items) <= limit:
            return items
        return items[-limit:]
    def iter_policies(self, limit: int = 100) -> list[Dict[str, Any]]:
        if not self.policy_path.exists():
            return []
        items: list[Dict[str, Any]] = []
        try:
            with self.policy_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return items[-limit:]
