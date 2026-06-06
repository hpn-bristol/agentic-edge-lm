"""LLM-driven routing policy planner and orchestrator.

When executed as a script, this module runs both:

1. A periodic LLM-based planner that updates the global routing policy every
   `ttl_minutes` slots.
2. A router loop that executes one environment step per slot using the latest
   policy plus short-window feedback, together with Lyapunov local control on
   each node.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import pathlib
import math
import random
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # Optional dependency – gracefully degrade if OpenAI SDK is absent
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore

from kubernetes import config as k8s_config, client as k8s_client
from pymongo import MongoClient
import sys

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent                 # .../MA/ctdema
REPO_ROOT = CURRENT_DIR.parents[2]                # .../ELLM
STATE_ROOT = pathlib.Path(os.getenv("ELLM_STATE_DIR", str(REPO_ROOT / "runtime_state")))
POLICY_DIR = pathlib.Path(os.getenv("ELLM_POLICY_DIR", str(STATE_ROOT / "llm_policy")))
sys.path.append(str(REPO_ROOT))                  # so MA.* imports work from repo root
sys.path.append(str(PROJECT_ROOT / "runtime"))
sys.path.append(str(PROJECT_ROOT / "llm-route"))  # for online_corrector / policy_store


def _load_env_from_file(path: pathlib.Path) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    Values from the OS take precedence; this is only a convenience so that
    running policy_manager.py directly from the repository root picks up settings
    from .env (e.g. OPENAI_API_KEY, LLM_POLICY_MODEL) without manual export.
    """
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best-effort helper; never break orchestrator if .env is malformed.
        pass


# Auto-load repo-level .env before reading env vars.
_load_env_from_file(REPO_ROOT / ".env")


def _slim_aggregate(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Keep key aggregate fields per anchor for history/meta (mean + coverage)."""
    slim: Dict[str, Any] = {}
    for aid, stats in (aggregate or {}).items():
        stats = stats or {}
        slim[aid] = {
            "mean_latency": float(stats.get("mean_latency", 0.0)),
            "valid_ratio": float(stats.get("valid_ratio", 0.0)),
        }
    return slim


def _load_macro_policy() -> Dict[str, Any]:
    """Load the current macro (high-level) strategy JSON, or return a default.

    The macro policy encodes anchor-level spreading/home patterns and
    per-node layout intent. It is produced by a separate high-level planner.
    """
    try:
        if MACRO_POLICY_PATH.exists():
            return json.loads(MACRO_POLICY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass

    # Conservative default: no special home; spread all anchors across all nodes,
    # and let local agents decide deployment.
    default_policy: Dict[str, Any] = {
        "epoch_id": 0,
        "anchor_policies": {
            "1": {"cpu_spread": {"fraction": 1.0, "targets": "all_nodes"}},
            "2": {"cpu_spread": {"fraction": 1.0, "targets": "all_nodes"}},
            "3": {"gpu_focus": {"fraction": 1.0, "targets": "gpu_nodes"}},
            "4": {"gpu_focus": {"fraction": 1.0, "targets": "gpu_nodes"}},
        },
        "node_layout_intent": {},
    }
    return default_policy

from MA.ctdema.llm.cluster_info import build_static_context_text, fetch_cluster_static_info  # type: ignore
from MA.ctdema.llm.monitor.history import HistoryStore  # type: ignore
from MA.ctdema.llm.monitor.snapshot import SnapshotBuilder, DEFAULT_MONGO_URI  # type: ignore
from MA.ctdema.llm.monitor.request_summary import build_request_summary  # type: ignore
from MA.ctdema.llm.agent_event_logger import AgentEventLogger, get_openai_usage  # type: ignore

from EllmEnv import EllmEnv  # type: ignore
from online_corrector import apply_online_correction  # type: ignore
from policy_store import PolicyStore  # type: ignore


# Request range used when constructing a short-lived EllmEnv helper for
# cluster-wide control operations (e.g. resetting local agent mode). The
# exact values do not matter for /set_agent_mode; we keep them consistent
# with other routing entrypoints.
REQUEST_RANGE = [[0, 8], [0, 8], [0, 8], [0, 8]]


def _parse_json_loose(raw: Optional[str]) -> Dict[str, Any]:
    """Best-effort JSON parser that extracts the first JSON object from raw text."""

    if not raw:
        return {}
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return {}
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass
    # Second attempt: replace single quotes and trailing commas.
    cleaned = snippet.replace("'", '"')
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace(",}", "}")
    cleaned = cleaned.replace(",]", "]")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


DEFAULT_POLICY_PATH = POLICY_DIR / "current_policy.json"
MACRO_POLICY_PATH = POLICY_DIR / "macro_policy.json"
MACRO_HISTORY_PATH = POLICY_DIR / "macro_history.jsonl"
MICRO_HISTORY_PATH = POLICY_DIR / "micro_history.jsonl"

SLOT_PERIOD_SEC = 30


def _resolve_optional_path(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        n = len(weights)
        if n == 0:
            return {}
        uniform = 1.0 / n
        return {k: uniform for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


@dataclass
class PolicyManagerConfig:
    model: str = os.getenv("LLM_POLICY_MODEL", "gpt-5")
    policy_path: pathlib.Path = DEFAULT_POLICY_PATH
    # Number of logical slots for which a generated route policy remains valid.
    ttl_minutes: int = int(os.getenv("LLM_POLICY_TTL_MIN", "1"))
    delta_limit: float = float(os.getenv("LLM_POLICY_DELTA_MAX", "0.1"))
    # Macro-level evaluation horizon and inner scoring window, both in slots.
    macro_epoch_slots: int = int(os.getenv("LLM_MACRO_EPOCH_SLOTS", "100"))
    macro_eval_prefix_slots: int = int(os.getenv("LLM_MACRO_EVAL_PREFIX", "50"))
    macro_eval_warmup_slots: int = int(os.getenv("LLM_MACRO_EVAL_WARMUP", "20"))
    exploration_dir: Optional[pathlib.Path] = (
        _resolve_optional_path(os.getenv("LLM_MACRO_EXPLORATION_DIR", "MA/ctdema/exploration"))
        if os.getenv("LLM_MACRO_EXPLORATION_DIR", "MA/ctdema/exploration").strip()
        else None
    )


class PolicyManager:
    """High level orchestrator for policy generation and persistence."""

    def __init__(self, config: Optional[PolicyManagerConfig] = None):
        self.config = config or PolicyManagerConfig()
        self.config.policy_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a fixed short window for snapshot-building; macro scoring no
        # longer depends on this rolling view.
        self.snapshot_builder = SnapshotBuilder()
        history_root = self.config.policy_path.parent / ".." / "llm_history"
        self.history = HistoryStore(base_path=history_root.resolve())
        # Macro-level tracking: epoch index and slot range.
        self.macro_history_path = MACRO_HISTORY_PATH
        self.macro_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_epoch_id: int = 0
        self._current_epoch_start_slot: Optional[int] = None
        # Marks whether the macro LLM has produced the first epoch policy.
        self._bootstrap_planned: bool = False
        # Logical planner tick counters: one tick per run_once invocation,
        # plus a per-epoch counter so that epoch progression does not depend
        # on telemetry latest_slot catching up.
        self._total_ticks: int = 0
        self._ticks_into_epoch: int = 0
        self._exploration_policies: List[pathlib.Path] = self._load_exploration_policy_paths()
        self._exploration_stage: int = 0
        # Best-effort structured logger for macro LLM invocations.
        self._event_logger = AgentEventLogger("macro")
        # Filled in by _call_llm so that downstream parse / validation
        # outcomes can be associated with the same call's telemetry.
        self._last_call_meta: Dict[str, Any] = {}

    def _load_exploration_policy_paths(self) -> List[pathlib.Path]:
        root = self.config.exploration_dir
        if root is None or not root.exists() or not root.is_dir():
            return []
        return sorted(p for p in root.glob("*.json") if p.is_file())

    def _next_exploration_policy(self) -> Optional[Dict[str, Any]]:
        if self._exploration_stage >= len(self._exploration_policies):
            return None
        path = self._exploration_policies[self._exploration_stage]
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[macro] WARNING: failed to load exploration policy {path.name}: {exc}")
            self._exploration_stage += 1
            return None
        self._exploration_stage += 1
        if not isinstance(obj, dict):
            print(f"[macro] WARNING: exploration policy {path.name} is not a JSON object.")
            return None
        print(f"[macro] Loaded exploration macro policy from {path.name}.")
        return obj

    def _plan_macro_policy_with_llm(self, latest_slot_int: int, *, initial: bool, epoch_start: Optional[int] = None, eval_until: Optional[int] = None) -> Dict[str, Any]:
        static_context = build_static_context_text()
        history_all = self._load_macro_history(limit=100)
        history = self._select_macro_history_for_prompt(history_all)
        prompt = self._compose_macro_prompt(
            latest_slot=latest_slot_int,
            static_context=static_context,
            macro_history=history,
        )

        # Persist the latest macro LLM input prompt for inspection.
        try:
            input_path = POLICY_DIR / "macro_input.json"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"generated_at": _now_utc_iso(), "prompt": prompt}
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        if initial:
            print(f"[macro] Calling macro LLM for initial epoch (latest_slot={latest_slot_int}).")
        else:
            print(
                f"[macro] Calling macro LLM for next epoch (current_epoch={self._current_epoch_id}, "
                f"latest_slot={latest_slot_int}, window=[{epoch_start}, {eval_until}])."
            )
        raw = self._call_llm(prompt)
        obj = _parse_json_loose(raw)
        if not isinstance(obj, dict) or not obj:
            self._log_macro_outcome("parse_fail", raw=raw)
            return {}
        self._log_macro_outcome("success")
        return obj

    # ------------------------------------------------------------------
    # Near-real-time global state summary (non-latency signals)
    # ------------------------------------------------------------------
    def _build_realtime_state_summary(self) -> str:
        """Summarise the most recent cluster state (CPU/memory/GPU and LM placement).

        This uses the latest k8smonitor snapshot and current pod layout to
        provide the LLM with a near-real-time view of resource headroom and
        model deployment, without relying on short-window latency.
        """
        dbname = os.getenv("LLM_ROUTING_DBNAME")
        if not dbname:
            return ""

        # Static capacities (including our logical GPU replica overrides).
        try:
            static_nodes = fetch_cluster_static_info()
        except Exception:
            static_nodes = {}

        # Latest resource metrics from k8smonitor.<dbname>
        metrics: Dict[str, Any] = {}
        # Resolve Mongo URI in the same way as SnapshotBuilder so that we
        # do not depend solely on LLM_MONGO_URI being set.
        try:
            mongo_uri = DEFAULT_MONGO_URI
            if not mongo_uri:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1_disc = k8s_client.CoreV1Api()
                svc = v1_disc.read_namespaced_service(
                    name="mongodb-service",
                    namespace="mongodb",
                )
                host = svc.spec.cluster_ip
                mongo_uri = f"mongodb://root:secret@{host}:27017"
            with MongoClient(mongo_uri) as client:
                col = client["k8smonitor"][dbname]
                doc = col.find_one(sort=[("timestamp", -1)])
                if doc:
                    metrics = doc.get("metrics", {}) or {}
        except Exception:
            metrics = {}

        # Remaining GPU units from ConfigMap (if available).
        gpu_remaining: Dict[str, int] = {}
        try:
            v1 = k8s_client.CoreV1Api()
            cm = v1.read_namespaced_config_map(name="remaining-gpu", namespace="allworkers")
            data_dict = cm.data or {}
            for node_name, val in data_dict.items():
                try:
                    gpu_remaining[node_name] = max(0, int(val))
                except Exception:
                    continue
        except Exception:
            gpu_remaining = {}

        # Current LM placement: which anchors are active on each node.
        active_lms: Dict[str, List[int]] = {}
        try:
            v1 = k8s_client.CoreV1Api()
            for info in static_nodes.values():
                node_name = info.name
                active_flags = [0, 0, 0, 0]
                try:
                    pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}")
                except Exception:
                    pods = None
                if pods:
                    for pod in pods.items:
                        labels = pod.metadata.labels or {}
                        llm_label = labels.get("llm") or ""
                        if not llm_label:
                            continue
                        # Support multiple label styles:
                        #  - "llm_1", "llm_2", ...
                        #  - "llm1-worker1", "llm2-worker3", ...
                        anchor_id: Optional[int] = None
                        if llm_label.startswith("llm_"):
                            tail = llm_label.split("_", 1)[1]
                        elif llm_label.startswith("llm"):
                            tail = llm_label[3:]
                        else:
                            tail = ""
                        if tail:
                            # Take the first digit in the tail as anchor id.
                            for ch in tail:
                                if ch.isdigit():
                                    try:
                                        anchor_id = int(ch)
                                    except Exception:
                                        anchor_id = None
                                    break
                        if anchor_id is not None and 1 <= anchor_id <= 4:
                            active_flags[anchor_id - 1] = 1
                active_lms[node_name] = active_flags
        except Exception:
            active_lms = {}

        # Build human-readable summary.
        lines: List[str] = []
        for ip, info in static_nodes.items():
            node_name = info.name
            m = metrics.get(node_name, {}) or {}
            try:
                cpu_util = float(m.get("cpu", 0.0))
            except Exception:
                cpu_util = 0.0
            cpu_free = max(0.0, 1.0 - cpu_util / 100.0)

            try:
                total_mb = float(m.get("total_memory", 0.0))
                free_mb = float(m.get("free_memory", 0.0)) / 1024.0
                mem_free = free_mb / total_mb if total_mb > 0 else 1.0
            except Exception:
                mem_free = 1.0

            rem_units = gpu_remaining.get(node_name)
            gpu_cap = info.gpu_replicas or 0
            if rem_units is not None and gpu_cap > 0:
                gpu_free = max(0.0, min(1.0, float(rem_units) / float(gpu_cap)))
                gpu_text = f"{gpu_free:.2f} ({rem_units}/{gpu_cap} units)"
            elif gpu_cap > 0:
                gpu_text = f"unknown (capacity={gpu_cap} units)"
            else:
                gpu_text = "N/A (no GPUs)"

            flags = active_lms.get(node_name, [0, 0, 0, 0])
            # Explicitly explain the status vector so the LLM knows the meaning.
            status_vec = "[" + ", ".join(str(int(f)) for f in flags) + "]"

            lines.append(
                f"- Node {node_name} ({ip}): cpu_free≈{cpu_free:.2f}, "
                f"mem_free≈{mem_free:.2f}, gpu_free={gpu_text}. "
                f"LM 1–4 activation status on this node is {status_vec} (0=inactive, 1=active)."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Macro-level evaluation over a fixed window
    # ------------------------------------------------------------------
    def _score_epoch(
        self,
        start_slot: int,
        eval_until_slot: int,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for an epoch over a fixed slot window.

        We aggregate per-LM mean latencies and success ratios from raw
        Mongo latency over slots [start_slot, eval_until_slot). Successful
        samples are defined as:
          - 0 < latency < 900, and
          - fail_reason is not an explicit timeout.

        From these we derive:
          - per_anchor_latency: epoch-mean latency per LM (success-only),
          - global_mean_latency: average of the four LMs' epoch means,
          - per_anchor_success_ratio: success / total per LM within the window,
          - jain_fairness: Jain's fairness index over per-LM success ratios,
          - eval_cost: a scalar objective defined as
                (global_mean_latency / 900) + (1 - jain_fairness),
            where lower is better.
        """

        # ------------------------------------------------------------------
        # Fetch raw latency from Mongo for this epoch prefix.
        # ------------------------------------------------------------------
        try:
            mongo_uri = getattr(self.snapshot_builder, "mongo_uri", DEFAULT_MONGO_URI)
        except Exception:
            mongo_uri = DEFAULT_MONGO_URI
        if not mongo_uri:
            return {
                "start_slot": int(start_slot),
                "eval_until_slot": int(eval_until_slot),
                "per_anchor_latency": {},
                "global_mean_latency": 0.0,
                "per_anchor_success_ratio": {},
                "jain_fairness": 0.0,
                "eval_cost": 0.0,
            }

        try:
            latency_db_name = getattr(self.snapshot_builder, "latency_db", "latency")
        except Exception:
            latency_db_name = "latency"

        anchors = ["1", "2", "3", "4"]
        # per-anchor -> slot -> list[latency] (successful samples only)
        per_anchor_slot_vals: Dict[str, Dict[int, List[float]]] = {
            aid: {} for aid in anchors
        }
        # Per-anchor success / total counts over the eval window so that we
        # can expose a success_service_ratio metric alongside mean latency.
        per_anchor_success: Dict[str, int] = {aid: 0 for aid in anchors}
        per_anchor_total: Dict[str, int] = {aid: 0 for aid in anchors}
        global_success = 0
        global_total = 0

        def _parse_slot(val: Any) -> Optional[int]:
            try:
                return int(str(val).strip())
            except Exception:
                return None

        try:
            with MongoClient(mongo_uri) as client:
                db = client[latency_db_name]
                for coll_name in db.list_collection_names():
                    coll = db[coll_name]
                    try:
                        # Do not rely on Mongo comparing slot_anchor as
                        # integers, since the field may be stored as a
                        # string. Fetch recent docs and filter by slot
                        # range in Python instead.
                        cursor = coll.find(
                            {},
                            {
                                "_id": 0,
                                "slot_anchor": 1,
                                "1": 1,
                                "2": 1,
                                "3": 1,
                                "4": 1,
                            },
                        )
                    except Exception:
                        continue
                    for doc in cursor:
                        slot = _parse_slot(doc.get("slot_anchor"))
                        if slot is None:
                            continue
                        if slot < int(start_slot) or slot >= int(eval_until_slot):
                            continue
                        for aid in anchors:
                            a_doc = doc.get(aid) or {}
                            if not isinstance(a_doc, dict) or not a_doc:
                                continue
                            # Explicitly skip records marked as skipped.
                            if a_doc.get("skip"):
                                continue
                            # Every non-skipped record within the eval window
                            # counts towards the denominator for success
                            # ratios, regardless of whether it ultimately
                            # completed successfully.
                            per_anchor_total[aid] += 1
                            global_total += 1

                            lat = a_doc.get("single_llm_delay")
                            fail_reason = a_doc.get("fail_reason")
                            if lat is None:
                                # No completed latency yet within the eval window:
                                # treat as unsuccessful (counted in totals but
                                # not in successful samples) so that success
                                # ratios reflect both hard failures and
                                # unfinished requests.
                                continue
                            try:
                                lat_val = float(lat)
                            except Exception:
                                continue
                            # Only keep successful samples:
                            #  - positive latency;
                            #  - strictly below the timeout-style 900s;
                            #  - fail_reason is not an explicit timeout.
                            if lat_val <= 0.0 or lat_val >= 900.0:
                                continue
                            if isinstance(fail_reason, str) and fail_reason.lower() == "timeout":
                                continue
                            # Successful sample.
                            per_anchor_success[aid] += 1
                            global_success += 1
                            slot_map = per_anchor_slot_vals[aid].setdefault(slot, [])
                            slot_map.append(lat_val)
        except Exception:
            # Fall through to empty per_anchor_slot_vals; scores will be 0.
            pass

        # ------------------------------------------------------------------
        # Aggregate per-anchor epoch mean latencies and success ratios.
        # ------------------------------------------------------------------
        per_anchor_latency: Dict[str, float] = {}
        per_anchor_success_ratio: Dict[str, float] = {}
        latencies: List[float] = []

        for aid in anchors:
            slot_map = per_anchor_slot_vals.get(aid, {})
            slot_means: List[float] = []
            for vals in slot_map.values():
                if not vals:
                    continue
                slot_means.append(float(sum(vals) / float(len(vals))))

            if not slot_means:
                # No successful latency samples for this LM in the eval
                # window: treat as a worst-case epoch mean.
                epoch_mean = 900.0
            else:
                epoch_mean = float(sum(slot_means) / float(len(slot_means)))
            per_anchor_latency[aid] = epoch_mean
            latencies.append(epoch_mean)

            # Success ratio for this LM over the eval window.
            tot = per_anchor_total.get(aid, 0)
            succ = per_anchor_success.get(aid, 0)
            per_anchor_success_ratio[aid] = float(succ) / float(tot) if tot > 0 else 0.0

        global_mean_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0

        # Jain's fairness index over per-LM success ratios:
        #   F = (sum_i rho_i)^2 / (I * sum_i rho_i^2),
        # where rho_i are per-LM success ratios and I is the number of LMs.
        succ_vals: List[float] = [float(per_anchor_success_ratio.get(aid, 0.0)) for aid in anchors]
        I = len(succ_vals)
        sum_r = float(sum(succ_vals))
        sum_sq = float(sum(v * v for v in succ_vals))
        if I <= 0 or sum_sq <= 0.0:
            jain_fairness = 0.0
        else:
            jain_fairness = (sum_r * sum_r) / (float(I) * sum_sq)

        # Combine latency and fairness into a single scalar cost:
        #   eval_cost = (global_mean_latency / 900) + (1 - jain_fairness),
        # where lower is better (shorter latency and more balanced success).
        norm_lat = (global_mean_latency / 900.0) if global_mean_latency > 0.0 else 0.0
        eval_cost = norm_lat + (1.0 - jain_fairness)

        # We only keep per-anchor/global latencies, success ratios, fairness,
        # and the scalar eval_cost here; epoch/slot indices are tracked
        # separately and are not required for macro LLM reasoning.
        # Per-LM workload observability: the TRUE per-LM offered load over this
        # eval window, read from the `workload` collection that EllmEnv
        # persists per slot. NOTE: per_anchor_total counts latency cells and
        # saturates at num_slots*num_nodes, so it is NOT a valid workload-volume
        # signal -- it is kept only as the success-ratio denominator above.
        per_anchor_request_count: Dict[str, int] = {aid: 0 for aid in anchors}
        try:
            run_dbname = os.getenv("LLM_ROUTING_DBNAME", "").strip()
            if run_dbname:
                with MongoClient(mongo_uri) as wclient:
                    wcoll = wclient["workload"][run_dbname]
                    for wdoc in wcoll.find(
                        {"slot_anchor": {"$gte": int(start_slot),
                                         "$lt": int(eval_until_slot)}},
                        {"_id": 0, "per_lm": 1},
                    ):
                        per_lm = wdoc.get("per_lm") or {}
                        for aid in anchors:
                            try:
                                per_anchor_request_count[aid] += int(per_lm.get(aid, 0))
                            except Exception:
                                pass
        except Exception:
            pass
        global_request_count = int(sum(per_anchor_request_count.values()))
        per_anchor_request_share: Dict[str, float] = {}
        for aid in anchors:
            per_anchor_request_share[aid] = (
                float(per_anchor_request_count[aid]) / float(global_request_count)
                if global_request_count > 0 else 0.0
            )

        return {
            "per_anchor_latency": per_anchor_latency,
            "global_mean_latency": global_mean_latency,
            "per_anchor_success_ratio": per_anchor_success_ratio,
            "jain_fairness": jain_fairness,
            "eval_cost": eval_cost,
            "per_anchor_request_count": per_anchor_request_count,
            "global_request_count": global_request_count,
            "per_anchor_request_share": per_anchor_request_share,
        }

    def _append_macro_history(self, record: Dict[str, Any]) -> None:
        try:
            with self.macro_history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Macro history is best-effort; never break orchestration.
            pass

    def _load_macro_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.macro_history_path.exists():
            return []
        items: List[Dict[str, Any]] = []
        try:
            with self.macro_history_path.open("r", encoding="utf-8") as f:
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

    def _select_macro_history_for_prompt(
        self,
        history: List[Dict[str, Any]],
        recent_k: int = 5,
        best_k: int = 3,
        worst_k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Select three labeled subsets of history records for the prompt.

        Returns a dict with three keys, each a chronological list:
          - "recent": the last ``recent_k`` epochs (immediate trend so the
            planner can tell whether the current policy is stable,
            improving, or drifting);
          - "best": up to ``best_k`` lowest-eval_cost epochs (anchor of
            "what good looks like" so the planner does not forget the best
            operating point reached so far);
          - "worst": up to ``worst_k`` highest-eval_cost epochs (negative
            precedents to avoid, per the paper's "learning from negative
            precedents" description).

        A given epoch may appear in more than one section (e.g. the most
        recent epoch is also the worst observed) — duplicating it across
        sections is intentional and reinforces the signal for the planner.
        """
        empty = {"recent": [], "best": [], "worst": []}
        n = len(history)
        if n <= 0:
            return empty

        recent_idx = sorted(range(max(0, n - recent_k), n)) if recent_k > 0 else []

        scored: List[tuple[int, float]] = []
        if best_k > 0 or worst_k > 0:
            for idx, rec in enumerate(history):
                score_block = rec.get("score", {}) or {}
                try:
                    s = float(score_block.get("eval_cost", 0.0))
                except Exception:
                    s = 0.0
                scored.append((idx, s))

        best_idx = (
            sorted(idx for idx, _ in sorted(scored, key=lambda x: x[1])[:best_k])
            if best_k > 0
            else []
        )
        worst_idx = (
            sorted(idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:worst_k])
            if worst_k > 0
            else []
        )

        return {
            "recent": [history[i] for i in recent_idx],
            "best": [history[i] for i in best_idx],
            "worst": [history[i] for i in worst_idx],
        }

    def _summarise_micro_epoch(
        self,
        epoch_id: int,
        window_offset: Optional[int] = None,
        window_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aggregate micro-level off-role routing statistics for a given epoch.

        We read micro_history.jsonl produced by the micro router, filter
        records with the given epoch_id, and aggregate:
          - total off_role_routes vs total_routes;
          - per-node off_role vs total and ratios.

        If window_offset/window_length are provided, we first infer a
        per-epoch slot index for each record as
            rel_slot = slot_index - min_slot_index_for_this_epoch
        and only include records with
            window_offset <= rel_slot < window_offset + window_length.
        This lets us align micro_off_role_ratio with the same “LLM-only”
        evaluation window used for latency scoring within each macro
        epoch, instead of averaging over the entire epoch including
        random-exploration segments.
        """
        if not MICRO_HISTORY_PATH.exists():
            return {"epoch_id": epoch_id}

        # First pass: collect all records for this epoch and find the
        # minimum slot_index so we can derive per-epoch relative slots.
        epoch_records: List[Dict[str, Any]] = []
        min_slot_index: Optional[int] = None
        try:
            with MICRO_HISTORY_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if int(rec.get("epoch_id", -1)) != epoch_id:
                        continue
                    epoch_records.append(rec)
                    try:
                        sidx_raw = rec.get("slot_index", None)
                        if sidx_raw is None:
                            continue
                        sidx = int(sidx_raw)
                    except Exception:
                        continue
                    if min_slot_index is None or sidx < min_slot_index:
                        min_slot_index = sidx
        except Exception:
            return {"epoch_id": epoch_id}

        if not epoch_records:
            return {"epoch_id": epoch_id}

        # Derive absolute window bounds in terms of slot_index for this
        # epoch, if a window has been requested.
        if window_offset is not None and window_length is not None and min_slot_index is not None:
            start_rel = max(0, int(window_offset))
            length_rel = max(0, int(window_length))
            start_abs = min_slot_index + start_rel
            end_abs = start_abs + length_rel
        else:
            # No windowing: include all slots for this epoch.
            start_abs = None
            end_abs = None

        # Second pass over the cached records: aggregate within the
        # requested window (or entire epoch when no window is given).
        total_off = 0.0
        total_routes = 0.0
        slots = 0
        per_node_acc: Dict[str, Dict[str, float]] = {}

        for rec in epoch_records:
            try:
                sidx_raw = rec.get("slot_index", None)
                sidx = int(sidx_raw) if sidx_raw is not None else None
            except Exception:
                sidx = None
            if start_abs is not None and end_abs is not None and sidx is not None:
                if sidx < start_abs or sidx >= end_abs:
                    continue
            slots += 1
            try:
                off = float(rec.get("off_role_routes", 0.0))
            except Exception:
                off = 0.0
            try:
                tot = float(rec.get("total_routes", 0.0))
            except Exception:
                tot = 0.0
            total_off += off
            total_routes += tot
            per_node = rec.get("per_node") or {}
            if isinstance(per_node, dict):
                for node, stats in per_node.items():
                    if not isinstance(stats, dict):
                        continue
                    try:
                        n_off = float(stats.get("off_role", 0.0))
                    except Exception:
                        n_off = 0.0
                    try:
                        n_tot = float(stats.get("total", 0.0))
                    except Exception:
                        n_tot = 0.0
                    acc = per_node_acc.setdefault(node, {"off_role": 0.0, "total": 0.0})
                    acc["off_role"] += n_off
                    acc["total"] += n_tot

        per_node_summary: Dict[str, Dict[str, float]] = {}
        for node, acc in per_node_acc.items():
            off = acc.get("off_role", 0.0)
            tot = acc.get("total", 0.0)
            ratio = (off / tot) if tot > 0.0 else 0.0
            per_node_summary[node] = {
                "off_role_routes": float(off),
                "total_routes": float(tot),
                "off_role_ratio": float(ratio),
            }
        off_ratio = (total_off / total_routes) if total_routes > 0.0 else 0.0
        return {
            "epoch_id": epoch_id,
            "slots_with_data": slots,
            "off_role_routes": float(total_off),
            "total_routes": float(total_routes),
            "off_role_ratio": float(off_ratio),
            "per_node": per_node_summary,
        }

    # ------------------------------------------------------------------
    # Snapshot + prompt preparation
    # ------------------------------------------------------------------
    def build_snapshot(self) -> Dict[str, Any]:
        # Build a short-window snapshot from Mongo for real-time context,
        # but do not persist it to snapshot_history.jsonl. Epoch-level
        # scoring now reads raw latency directly from Mongo instead of
        # relying on these rolling-window aggregates.
        snapshot = self.snapshot_builder.build()
        return snapshot

    @staticmethod
    def _macro_history_record_to_json(rec: Dict[str, Any]) -> Optional[str]:
        """Render a single macro_history record into the compact JSON form
        shown to the planner. Returns None on malformed inputs."""
        try:
            score_block = rec.get("score", {}) or {}
            gmean = score_block.get("global_mean_latency")
            gfair = score_block.get("jain_fairness")
            per_anchor_lat = score_block.get("per_anchor_latency") or {}
            per_anchor_succ = score_block.get("per_anchor_success_ratio") or {}
            per_anchor_cnt = score_block.get("per_anchor_request_count") or {}
            per_anchor_shr = score_block.get("per_anchor_request_share") or {}
            micro = rec.get("micro_summary", {}) or {}
            off_ratio = micro.get("off_role_ratio")
            macro = rec.get("macro_policy", {}) or {}
            item: Dict[str, Any] = {}
            if isinstance(gmean, (int, float)):
                item["global_mean_latency"] = round(float(gmean), 2)
            if isinstance(gfair, (int, float)):
                item["jain_fairness"] = round(float(gfair), 3)
            if isinstance(per_anchor_lat, dict) and per_anchor_lat:
                try:
                    item["per_LM_mean_latency"] = {
                        str(aid): round(float(val), 2)
                        for aid, val in per_anchor_lat.items()
                        if isinstance(val, (int, float))
                    }
                except Exception:
                    pass
            if isinstance(per_anchor_succ, dict) and per_anchor_succ:
                try:
                    item["per_LM_success_ratio"] = {
                        str(aid): round(float(val), 3)
                        for aid, val in per_anchor_succ.items()
                        if isinstance(val, (int, float))
                    }
                except Exception:
                    pass
            if isinstance(per_anchor_cnt, dict) and per_anchor_cnt:
                try:
                    item["per_LM_request_count"] = {
                        str(aid): int(val)
                        for aid, val in per_anchor_cnt.items()
                        if isinstance(val, (int, float))
                    }
                except Exception:
                    pass
            if isinstance(per_anchor_shr, dict) and per_anchor_shr:
                try:
                    item["per_LM_request_share"] = {
                        str(aid): round(float(val), 3)
                        for aid, val in per_anchor_shr.items()
                        if isinstance(val, (int, float))
                    }
                except Exception:
                    pass
            if isinstance(off_ratio, (int, float)):
                item["micro_off_role_ratio"] = round(float(off_ratio), 3)
            item["LM_policies"] = macro.get("LM_policies", {})
            item["node_layout_intent"] = macro.get("node_layout_intent", {})
            return json.dumps(item, ensure_ascii=False)
        except Exception:
            return None

    def _compose_macro_prompt(
        self,
        latest_slot: int,
        static_context: str,
        macro_history: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """Compose a compact prompt for the macro (high-level) strategy planner.

        The macro planner sees: (1) static cluster layout, (2) a slimmed
        compact history of previous macro policies with their scores. It must output a new macro_policy
        JSON (LM_policies + node_layout_intent), not concrete per-slot
        routes.
        """
        lines: List[str] = []
        lines.append(
            "You plan macro (epoch-level) traffic for a 4-LM, multi-node MEC inference cluster."
        )
        lines.append(
            "For the next epoch, choose high-level LM traffic splits and node roles; per-request routing and on-node deployment are handled by other agents."
        )
        lines.append(
            "The objective of your task is to reduce global average latency and improve fairness across LMs, as measured by the Jain's fairness index over per-LM success service ratios."
        )

        # ------------------------------------------------------------------
        # Static cluster context
        # ------------------------------------------------------------------
        lines.append("")
        # static_context already includes a header line and per-node capacities.
        lines.append(static_context)

        # ------------------------------------------------------------------
        # LM characteristics
        # ------------------------------------------------------------------
        lines.append("")
        lines.append("LM types (for intuition):")
        lines.append(
            "LM 1: latency-sensitive text, light-weight; GPU strongly preferred, CPU only as slow fallback."
        )
        lines.append(
            "LM 2: heavier latency-sensitive text; GPU strongly preferred, CPU even worse than LM1."
        )
        lines.append("LM 3: image generation, effectively GPU-only.")
        lines.append("LM 4: image understanding, effectively GPU-only.")
        lines.append(
            "Practical rule: all four LMs should primarily use GPUs; CPUs serve only as a small backup for LM1/LM2 on a few nodes."
        )

        # ------------------------------------------------------------------
        # Macro history (3 labeled sections: best / worst / recent)
        # ------------------------------------------------------------------
        lines.append("")
        lines.append(
            "Each epoch record below includes per_LM_request_count (number of "
            "requests of each LM type observed in that epoch's evaluation "
            "window) and per_LM_request_share (that LM type's fraction of total "
            "requests): together they describe the workload that policy "
            "operated under -- per_LM_request_count for total volume, "
            "per_LM_request_share for the workload type mix. Compare them "
            "across the epochs below to judge whether the workload is stable "
            "or shifting."
        )
        sections = [
            (
                "best",
                "Best historical epochs (policies that worked well in the "
                "past; each shows the per_LM_request_count / per_LM_request_share "
                "it operated under -- use them as reference, but a past policy "
                "is only a good template when the current workload is similar):",
            ),
            (
                "worst",
                "Worst historical epochs (negative precedents to avoid; "
                "do NOT reproduce these allocation patterns):",
            ),
            (
                "recent",
                "Most recent epochs in chronological order (read these as a "
                "trajectory: are things stable, improving, or drifting?):",
            ),
        ]
        any_records = isinstance(macro_history, dict) and any(
            macro_history.get(k) for k, _ in sections
        )
        if any_records:
            for key, header in sections:
                recs = macro_history.get(key, []) if isinstance(macro_history, dict) else []
                if not recs:
                    continue
                lines.append(f"=== {header} ===")
                for rec in recs:
                    rendered = self._macro_history_record_to_json(rec)
                    if rendered is not None:
                        lines.append(rendered)
                lines.append("")
        else:
            lines.append(
                "No previous macro history is available yet; you are designing the first macro strategy."
            )

        # ------------------------------------------------------------------
        # Instructions and output schema
        # ------------------------------------------------------------------
        lines.append("")
        lines.append(
            "Based on this history, design a new macro policy. Your objective is to create macro policies and node layouts that both reduce global_mean_latency and increase the Jain's fairness index over per-LM success ratios."
        )
        lines.append(
            "If workload mix and recent performance are stable, keep the current policy or make only small adjustments."
        )
        lines.append(
            "When an LM has higher recent request share and degraded latency or success ratio, increase its effective service capacity by revising both its traffic split and node-role assignment so that additional feasible service paths are available, instead of concentrating more traffic on the same destination set."
        )
        lines.append(
            "If recent micro_off_role_ratio is high in history, slightly relax node roles for overloaded LMs/nodes so that the micro router can stay on-role more often."
        )
        lines.append(
            "Your macro policy must define: (1) LM_policies (per-LM soft traffic split over node IPs), and (2) node_layout_intent (per-node allowed LMs on GPU)."
        )
        lines.append(
            "Consistency: if LM_policies[LM][ip] > 0 for a node IP, the corresponding worker's node_layout_intent.LMs list must contain that LM id."
        )
        lines.append(
            "The node_layout_intent schema must be: node_layout_intent[node_name] = {\"LMs\": [\"1\",\"2\",...]}."
        )
        lines.append("")
        lines.append("Return ONLY a strict JSON object with the following structure:")
        # example = {
        #     "epoch_id": "<int, next epoch id>",
        #     "LM_policies": {
        #         "1": {
        #             "<node-ip-a>": 0.2,
        #             "<node-ip-b>": 0.3,
        #             "<node-ip-c>": 0.5,
        #         },
        #         "2": {
        #             "<node-ip-a>": 0.3,
        #             "<node-ip-b>": 0.7,
        #         },
        #         "3": {
        #             "<node-ip-c>": 0.5,
        #             "<node-ip-d>": 0.5,
        #         },
        #         "4": {
        #             "<node-ip-d>": 0.7,
        #             "<node-ip-c>": 0.3,
        #         },
        #     },
        #     "node_layout_intent": {
        #         "worker1": {
        #             "LMs": ["1", "2"],
        #         },
        #         "worker2": {
        #             "LMs": ["1", "2"],
        #         },
        #         "worker3": {
        #             "LMs": ["3"],
        #         },
        #         "worker4": {
        #             "LMs": ["4"],
        #         },
        #         "worker5": {
        #             "LMs": ["1", "3", "4"],
        #         },
        #         "worker6": {
        #             "LMs": [],
        #         },
        #     },
        # }
        example = {
            "epoch_id": "<int, next epoch id>",
            "LM_policies": {
                "1": {
                    "<node-ip-a>": 0.2,
                    "<node-ip-b>": 0.3,
                    "<node-ip-c>": 0.3,
                    "<node-ip-d>": 0.2,
                },
                "2": {
                    "<node-ip-a>": 0.3,
                    "<node-ip-b>": 0.4,
                    "<node-ip-d>": 0.3,
                },
                "3": {
                    "<node-ip-e>": 0.5,
                    "<node-ip-c>": 0.3,
                    "<node-ip-f>": 0.2,
                },
                "4": {
                    "<node-ip-g>": 0.7,
                    "<node-ip-c>": 0.3,
                },
            },
            "node_layout_intent": {
                "worker1": {
                    "LMs": ["1", "2"],
                },
                "worker2": {
                    "LMs": ["1", "2"],
                },
                "worker3": {
                    "LMs": ["3"],
                },
                "worker4": {
                    "LMs": ["4"],
                },
                "worker5": {
                    "LMs": ["1", "3", "4"],
                },
                "worker6": {
                    "LMs": [],
                },
                "worker7": {
                    "LMs": ["3"],
                },
                "worker8": {
                    "LMs": ["1", "2", "4"],
                },
            },
        }
        lines.append(json.dumps(example, ensure_ascii=False, indent=2))
        lines.append(
            "You may add extra fields if helpful, but keep the policy easy for another LLM to interpret. Do NOT include any natural-language explanation, only the JSON object."
        )
        return "\n".join(lines)

    def _summarise_snapshot(self, snapshot: Dict[str, Any]) -> str:
        nodes = snapshot.get("nodes", [])
        anchors = snapshot.get("anchors", {})
        lines: List[str] = []
        meta = snapshot.get("meta", {})
        lines.append(
            f"Window={meta.get('window_slots', 'unknown')} slots, latest slot={meta.get('latest_slot', 'unknown')}"
        )
        if nodes:
            lines.append(
                "Per-node statistics below are keyed by origin node: each 'Node X' line summarises latency for requests that originated on node X, regardless of where they were actually processed."
            )
        for node in nodes:
            node_lines = [f"Node {node}:"]
            for anchor_id in sorted(anchors.keys()):
                node_metrics = anchors.get(anchor_id, {}).get(node)
                if not node_metrics:
                    continue
                lat = float(node_metrics.get("mean_latency", 0.0))
                p90 = float(node_metrics.get("p90_latency", 0.0))
                samples = int(node_metrics.get("samples", 0))
                node_lines.append(
                    "  LM {aid}: mean_latency={lat:.2f}s, p90_latency={p90:.2f}s, samples={s}".format(
                        aid=anchor_id,
                        lat=lat,
                        p90=p90,
                        s=samples,
                    )
                )
            if len(node_lines) > 1:
                lines.extend(node_lines)
        aggregate = snapshot.get("aggregate", {})
        if aggregate:
            lines.append("Aggregate (avg across nodes; per-LM coverage reflects how many node×slot combinations had completed latency in this window):")
            for anchor_id, aggr in aggregate.items():
                mean_lat = float(aggr.get("mean_latency", 0.0))
                timeout_ratio = float(aggr.get("ratio_900", 0.0))
                coverage = float(aggr.get("valid_ratio", 0.0))
                lines.append(
                    "  LM {aid}: mean_latency≈{lat:.2f}s, timeout_ratio≈{tr:.1%}, coverage≈{cov:.1%}".format(
                        aid=anchor_id,
                        lat=mean_lat,
                        tr=timeout_ratio,
                        cov=coverage,
                    )
                )
        req_summary = snapshot.get("request_summary") or {}
        if req_summary:
            window = req_summary.get("window_slots", self.config.window_slots)
            lines.append(f"Request distribution over last {window} slots:")
            per_node = req_summary.get("per_node", {})
            for node, info in per_node.items():
                total = info.get("total_prompts", 0.0)
                lines.append(f"Node {node}: total_prompts≈{total:.0f}")
                for anchor_id, anchor_info in (info.get("anchors") or {}).items():
                    lines.append(
                        "  LM {aid}: prompts≈{count:.0f} ({share:.1%})".format(
                            aid=anchor_id,
                            count=float(anchor_info.get("prompts", 0.0)),
                            share=float(anchor_info.get("share", 0.0)),
                        )
                    )
            aggr = req_summary.get("aggregate") or {}
            if aggr:
                lines.append("Overall prompt share:")
                for anchor_id, anchor_info in aggr.items():
                    lines.append(
                        "  LM {aid}: prompts≈{count:.0f} ({share:.1%})".format(
                            aid=anchor_id,
                            count=float(anchor_info.get("prompts", 0.0)),
                            share=float(anchor_info.get("share", 0.0)),
                        )
                    )
        return "\n".join(lines)

    def _vectorise_snapshot(self, snapshot: Dict[str, Any]) -> List[float]:
        """Vector used for memory retrieval.

        We deliberately base similarity on input-side conditions (request mix
        and effective load on nodes) rather than latency outcomes, so that we
        retrieve cases with similar workload / resource patterns, not
        necessarily similar realised performance.
        """
        req = snapshot.get("request_summary", {}) or {}
        req_aggr = req.get("aggregate", {}) or {}
        per_node = req.get("per_node", {}) or {}

    # ------------------------------------------------------------------
    # LLM call & parsing
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> Optional[str]:
        # In debug mode we prefer hard failures over silent fallbacks so that
        # environment / API issues are immediately visible.
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK is not available; install the 'openai' package to use the LLM planner.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call the LLM planner.")

        # Refresh dbname for the agent-event logger in case it was set after
        # this PolicyManager was instantiated (e.g. by main()).
        self._event_logger.set_dbname(os.getenv("LLM_ROUTING_DBNAME") or "")

        client = OpenAI()
        t_start = time.time()
        try:
            # Some models only support the default temperature. To stay
            # compatible, we omit the temperature argument and rely on the
            # model's default (typically 1.0).
            completion = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You synthesise routing policies for a distributed inference"
                            " cluster. Always respond with strict JSON (no prose)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            latency_ms = int((time.time() - t_start) * 1000)
            self._last_call_meta = {
                "latency_ms": latency_ms,
                "tokens_in": None,
                "tokens_out": None,
            }
            self._event_logger.log(
                "api_error",
                epoch_id=self._current_epoch_id,
                latency_ms=latency_ms,
                error=str(exc),
                model=self.config.model,
            )
            raise RuntimeError(f"LLM planner call failed: {exc}") from exc

        usage = get_openai_usage(completion)
        self._last_call_meta = {
            "latency_ms": int((time.time() - t_start) * 1000),
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
        }
        return completion.choices[0].message.content if completion.choices else None

    def _log_macro_outcome(
        self,
        outcome: str,
        raw: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
    ) -> None:
        """Emit a macro-layer agent event with the most recent call telemetry."""
        meta = self._last_call_meta or {}
        kwargs: Dict[str, Any] = {
            "epoch_id": self._current_epoch_id,
            "latency_ms": meta.get("latency_ms"),
            "tokens_in": meta.get("tokens_in"),
            "tokens_out": meta.get("tokens_out"),
            "model": self.config.model,
        }
        if outcome in ("parse_fail", "validation_fail"):
            kwargs["raw_truncated"] = raw
        if validation_errors:
            kwargs["validation_errors"] = validation_errors
        self._event_logger.log(outcome, **kwargs)

    def _fallback_policy(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        nodes: List[str] = list(snapshot.get("nodes", []) or [])
        anchors: List[str] = list(snapshot.get("anchor_ids", ["1", "2", "3", "4"]) or [])
        # Fallback: simple identity mapping per origin node (keep traffic local).
        routes = {src_ip: {aid: src_ip for aid in anchors} for src_ip in nodes}
        return {
            "routes": routes,
            "meta": {
                "generated_at": _now_utc_iso(),
                "ttl_slots": self.config.ttl_minutes,
                "delta_limit": self.config.delta_limit,
                "source": "fallback-uniform",
            },
        }

    def _parse_policy(self, raw: Optional[str], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not raw:
            return self._fallback_policy(snapshot)
        candidate = _parse_json_loose(raw)
        if not candidate.get("routes"):
            return self._fallback_policy(snapshot)
        return candidate

    # ------------------------------------------------------------------
    # Validation & persistence
    # ------------------------------------------------------------------
    def _validate_and_enrich(self, policy: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        nodes: List[str] = list(snapshot.get("nodes", []) or [])
        anchors: List[str] = list(snapshot.get("anchor_ids", ["1", "2", "3", "4"]) or [])

        raw_routes: Dict[str, Any] = policy.get("routes", {}) or {}
        routes: Dict[str, Any]

        # Detect anchor->probabilities format: {"1": {"ip": prob, ...}, ...}
        is_anchor_weight_format = bool(raw_routes) and all(
            k in anchors and isinstance(v, dict) for k, v in raw_routes.items()
        )
        # Detect new origin->anchor->dest_ip format: {"<origin-ip>": {"1": "<dest-ip>", ...}, ...}
        node_set = set(nodes)
        is_origin_format = bool(raw_routes) and all(
            (k in node_set) and isinstance(v, dict) for k, v in raw_routes.items()
        )

        if is_origin_format:
            # Normalise new format: ensure every origin node and anchor has a valid destination node.
            routes_by_origin: Dict[str, Dict[str, str]] = {}
            for src_ip in nodes:
                src_map = raw_routes.get(src_ip, {}) or {}
                norm_map: Dict[str, str] = {}
                for aid in anchors:
                    dst_ip = src_map.get(aid)
                    if isinstance(dst_ip, str) and dst_ip in node_set:
                        norm_map[aid] = dst_ip
                    else:
                        # Fallback: if missing or invalid, keep traffic local to the origin node when possible.
                        norm_map[aid] = src_ip if src_ip in node_set else (nodes[0] if nodes else "")
                routes_by_origin[src_ip] = norm_map
            routes = routes_by_origin
        elif is_anchor_weight_format:
            # Normalise anchor->probabilities to sum to 1 per anchor.
            routes_anchor: Dict[str, Dict[str, float]] = {}
            for aid in anchors:
                raw_weights = raw_routes.get(aid, {}) or {}
                weights = {node: float(raw_weights.get(node, 0.0)) for node in nodes}
                routes_anchor[aid] = _normalise(weights)
            routes = routes_anchor
        else:
            # Fallback: build a simple identity mapping per origin node to avoid crashes.
            routes = {src_ip: {aid: src_ip for aid in anchors} for src_ip in nodes}

        meta = policy.get("meta", {}).copy()
        meta.setdefault("generated_at", _now_utc_iso())
        # Force ttl_slots to follow the local config rather than whatever
        # the LLM happened to emit, so that planner frequency and router
        # refresh semantics stay aligned.
        meta["ttl_slots"] = max(1, self.config.ttl_minutes)
        meta.setdefault("delta_limit", self.config.delta_limit)
        meta.setdefault("version", policy.get("meta", {}).get("version", 1))
        meta.setdefault("source", policy.get("meta", {}).get("source", "llm"))

        enriched = {"routes": routes, "meta": meta}
        return enriched

    def save_policy(self, policy: Dict[str, Any]) -> None:
        payload = json.dumps(policy, ensure_ascii=False, indent=2)
        self.config.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.policy_path.write_text(payload, encoding="utf-8")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_once(self) -> Dict[str, Any]:
        # Treat each planner invocation as one logical slot/tick for epoch
        # bookkeeping so that macro planning progresses with the
        # orchestrator cadence instead of waiting for telemetry to
        # advance latest_slot.
        self._total_ticks += 1

        snapshot = self.build_snapshot()
        meta = snapshot.get("meta", {}) or {}
        latest_slot_raw = meta.get("latest_slot")
        try:
            latest_slot_int = int(latest_slot_raw)
        except Exception:
            latest_slot_int = 0

        epoch_len = max(1, self.config.macro_epoch_slots)
        eval_prefix = max(1, min(self.config.macro_epoch_slots, self.config.macro_eval_prefix_slots))
        eval_warmup = max(0, min(epoch_len - 1, int(getattr(self.config, "macro_eval_warmup_slots", 0))))

        # If no macro policy has been generated yet, call the macro LLM once
        # to plan the first epoch. This avoids an initial epoch driven only by
        # the built-in fallback policy.
        if not self._bootstrap_planned:
            obj_epoch = 1
            obj = self._next_exploration_policy()
            if obj is None:
                obj = self._plan_macro_policy_with_llm(latest_slot_int, initial=True)

            # Prefer the returned epoch id when available, falling back to 1.
            try:
                obj_epoch = int(obj.get("epoch_id", 1))
            except Exception:
                obj_epoch = 1
            obj["epoch_id"] = obj_epoch

            try:
                MACRO_POLICY_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[macro] Updated macro_policy.json for initial epoch {obj_epoch}.")
            except Exception as exc:
                print(f"[macro] WARNING: failed to write macro_policy.json for initial epoch: {exc}")

            # Align internal epoch state with the policy we just wrote.
            self._current_epoch_id = obj_epoch
            self._bootstrap_planned = True
            self._current_epoch_start_slot = 0
            self._ticks_into_epoch = 0

            # For compatibility, return a harmless routing-style policy for
            # the caller; micro-level routing reads macro_policy.json
            # directly.
            return self._fallback_policy(snapshot)

        # Past bootstrap and initial planning: advance epoch bookkeeping
        # based purely on planner ticks so that we do not block on late
        # latency completions. Each epoch covers macro_epoch_slots logical
        # ticks; scoring uses only the first eval_prefix slots.
        self._ticks_into_epoch += 1

        # Initialise epoch start slot label if missing (should normally be
        # set in the initial planning block above).
        if self._current_epoch_start_slot is None:
            self._current_epoch_start_slot = 0

        epoch_start = int(self._current_epoch_start_slot)

        # If we have completed at least one full epoch worth of planner
        # ticks since the last update, compute its score and append to
        # macro history, then plan the next one.
        if self._ticks_into_epoch >= epoch_len:
            eval_start = epoch_start + eval_warmup
            # Ensure evaluation window stays within the logical epoch.
            eval_until = min(epoch_start + epoch_len, eval_start + eval_prefix)
            score = self._score_epoch(eval_start, eval_until)
            # Align micro_off_role_ratio with the same logical window
            # used for latency scoring by passing the per-epoch warmup
            # and prefix slots as a relative window.
            micro_summary = self._summarise_micro_epoch(
                self._current_epoch_id,
                window_offset=eval_warmup,
                window_length=eval_prefix,
            )
            try:
                current_macro = _load_macro_policy()
            except Exception:
                current_macro = {}
            # Store only outcome and policy; internal epoch/slot labels are
            # kept in logs and micro_summary but omitted from history JSON
            # to keep the structure focused on action/reward.
            record = {
                "score": score,
                "micro_summary": micro_summary,
                "macro_policy": current_macro,
            }
            self._append_macro_history(record)
            eval_cost = float(score.get("eval_cost", 0.0))
            print(
                f"[macro] Epoch {self._current_epoch_id} scored "
                f"eval_cost≈{eval_cost:.3f} "
                f"over slots [{eval_start}, {eval_until}] (latest_slot={latest_slot_int})."
            )

            # Plan the next epoch with the macro LLM.
            next_epoch_id = self._current_epoch_id + 1
            obj = self._next_exploration_policy()
            if obj is None:
                obj = self._plan_macro_policy_with_llm(
                    latest_slot_int,
                    initial=False,
                    epoch_start=epoch_start,
                    eval_until=eval_until,
                )

            # Epoch id is a controller-side sequence counter, NOT something the
            # LLM gets to set. Force it to advance by exactly +1 each cycle.
            obj_epoch = next_epoch_id
            obj["epoch_id"] = obj_epoch

            # Save the new macro policy JSON.
            try:
                MACRO_POLICY_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[macro] Updated macro_policy.json for epoch {obj_epoch}.")
            except Exception as exc:
                print(f"[macro] WARNING: failed to write macro_policy.json: {exc}")

            # Advance epoch counters.
            self._current_epoch_id = obj_epoch
            # Next epoch starts immediately after the previous epoch's
            # nominal end slot; ticks into the new epoch are reset.
            self._current_epoch_start_slot = epoch_start + epoch_len
            self._ticks_into_epoch = 0

        # For compatibility, return a harmless routing-style policy, but
        # in the new architecture micro-level routing is handled elsewhere
        # using macro_policy.json directly.
        return self._fallback_policy(snapshot)


async def _router_loop(network: EllmEnv, policy_store: PolicyStore, window_slots: int) -> None:
    """Router loop: one step per slot, using latest policy + short-window feedback."""
    snapshot_builder = SnapshotBuilder(window_slots=window_slots)
    policy = policy_store.load()
    if not policy:
        # Uniform anchor-level bootstrap policy.
        policy = PolicyStore.build_uniform_policy(network.node_ips)
        policy_store.save(policy)
    policy = policy_store.stamp_loaded(policy)

    def _candidate_indices(env: EllmEnv) -> List[List[int]]:
        """Per-head candidate node indices, matching llm-route/router.py.

        Heads 0/1 (anchors 1/2): all nodes.
        Heads 2/3 (anchors 3/4): GPU nodes only (fall back to node 0 if none).
        """
        gpu_node_indices = [
            i
            for i, n in enumerate(env.nodes)
            if env.node_resources.get(n, {}).get("max_gpu", 0)
        ]
        if not gpu_node_indices:
            gpu_node_indices_local = [0]
        else:
            gpu_node_indices_local = gpu_node_indices
        all_indices = list(range(len(env.node_ips)))
        # head0->anchor1, head1->anchor2, head2->anchor3, head3->anchor4
        return [all_indices, all_indices, gpu_node_indices_local, gpu_node_indices_local]

    def _sample_action(
        routes: Dict[str, Dict[str, float]],
        candidates: List[List[int]],
        node_ips: List[str],
    ) -> List[int]:
        """Sample a 4-head action for anchors 1–4 using policy weights."""

        def _pick_for_anchor(anchor: str, head_idx: int) -> int:
            weights = routes.get(anchor, {})
            candidate_indices = candidates[head_idx]
            node_weights: List[float] = []
            for idx in candidate_indices:
                node_ip = node_ips[idx]
                node_weights.append(max(0.0, float(weights.get(node_ip, 0.0))))
            total = sum(node_weights)
            if total <= 0:
                node_weights = [1.0 / len(candidate_indices)] * len(candidate_indices)
            else:
                node_weights = [w / total for w in node_weights]

            try:
                eps = float(os.getenv("LLM_ROUTER_EPSILON", "0.05"))
            except Exception:
                eps = 0.0
            if eps > 0.0 and len(node_weights) > 0:
                u = 1.0 / len(node_weights)
                node_weights = [(1.0 - eps) * w + eps * u for w in node_weights]

            return int(random.choices(range(len(candidate_indices)), weights=node_weights, k=1)[0])

        # Initialise with zeros; fill heads for anchors 1–4.
        action: List[int] = [0, 0, 0, 0]
        # anchor 1 -> head 0
        action[0] = _pick_for_anchor("1", 0)
        # anchor 2 -> head 1
        action[1] = _pick_for_anchor("2", 1)
        # anchor 3 -> head 2
        action[2] = _pick_for_anchor("3", 2)
        # anchor 4 -> head 3
        action[3] = _pick_for_anchor("4", 3)
        return action

    def _direct_action_for_origin(
        routes_by_origin: Dict[str, Dict[str, str]],
        candidates: List[List[int]],
        node_ips: List[str],
        origin_index: int,
    ) -> List[int]:
        """Deterministic 4-head action for anchors 1–4 based on per-origin routing.

        routes_by_origin has the form {origin_ip: {anchor_id: target_ip}}.
        For each anchor/head we map the configured target_ip to an index
        within the corresponding candidate list; if the target is missing
        or invalid we fall back to the first candidate.
        """

        src_ip = node_ips[origin_index]
        mapping = routes_by_origin.get(src_ip, {}) or {}
        # Initialise with zeros; fill heads for anchors 1–4.
        action: List[int] = [0, 0, 0, 0]
        anchors_order = ["1", "2", "3", "4"]

        for head_idx, aid in enumerate(anchors_order):
            candidate_indices = candidates[head_idx]
            if not candidate_indices:
                action[head_idx] = 0
                continue
            dst_ip = mapping.get(aid)
            chosen_pos: Optional[int] = None
            if isinstance(dst_ip, str):
                for pos, global_idx in enumerate(candidate_indices):
                    if node_ips[global_idx] == dst_ip:
                        chosen_pos = pos
                        break
            if chosen_pos is None:
                chosen_pos = 0
            action[head_idx] = chosen_pos

        return action

    candidates = _candidate_indices(network)

    while True:
        start_ts = time.time()

        if policy_store.needs_refresh(policy):
            new_policy = policy_store.load()
            if new_policy:
                policy = policy_store.stamp_loaded(new_policy)
                print("[orchestrator] Loaded refreshed policy JSON.")
        try:
            # Still build a short-window snapshot to validate telemetry
            # pipeline and keep Mongo touch behaviour consistent, but do
            # not modify the LLM policy based on this noisy signal. The
            # router now follows the LLM routes directly (plus epsilon
            # exploration), without online correction.
            snapshot_builder.build()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[orchestrator] Snapshot build failed: {exc}")

        routes_any: Dict[str, Any] = policy.get("routes", {}) or {}
        actions: List[List[int]] = []

        # Detect whether we are using per-origin deterministic routing or anchor-level weights.
        node_set = set(network.node_ips)
        is_origin_format = bool(routes_any) and all(
            (k in node_set) and isinstance(v, dict) for k, v in routes_any.items()
        )

        if is_origin_format:
            routes_by_origin: Dict[str, Dict[str, str]] = {
                k: (v or {}) for k, v in routes_any.items()
            }
            for origin_index, _ in enumerate(network.nodes):
                actions.append(
                    _direct_action_for_origin(
                        routes_by_origin, candidates, network.node_ips, origin_index
                    )
                )
        else:
            # Anchor-level weighted routing: routes = {anchor: {node_ip: prob}}
            routes_anchor: Dict[str, Dict[str, float]] = {
                k: (v or {}) for k, v in routes_any.items()
            }
            for _ in network.nodes:
                actions.append(_sample_action(routes_anchor, candidates, network.node_ips))

        await network.step(actions)

        elapsed = time.time() - start_ts
        sleep_rem = max(0.0, SLOT_PERIOD_SEC - elapsed)
        if sleep_rem > 0:
            await asyncio.sleep(sleep_rem)


async def _planner_loop(mgr: PolicyManager) -> None:
    """Periodic planner loop: run LLM on a fixed slot-aligned schedule.

    We align planner triggers to a regular grid of slot boundaries defined by
    ttl_minutes * SLOT_PERIOD_SEC.  If a planner call takes longer than one
    interval, we apply its result as soon as it completes but schedule the
    next trigger at the next future boundary (we do not run planners in
    parallel or try to "catch up" missed intervals).
    """

    interval_sec = max(1, mgr.config.ttl_minutes) * SLOT_PERIOD_SEC
    start_time = time.time()
    next_run_at = start_time  # first run as soon as possible

    while True:
        now = time.time()
        sleep_rem = next_run_at - now
        if sleep_rem > 0:
            await asyncio.sleep(sleep_rem)

        print("[orchestrator] Running LLM planner once...")
        # Run the synchronous planner in a thread so we do not block the event loop.
        # Any exception here should surface and stop the orchestrator so that bugs
        # (e.g., missing API keys or bad responses) are clearly visible.
        await asyncio.to_thread(mgr.run_once)

        # After completion, schedule the next trigger at the next interval
        # boundary strictly after "now", staying aligned to the fixed grid.
        now = time.time()
        elapsed = max(0.0, now - start_time)
        intervals_passed = math.floor(elapsed / interval_sec)
        next_run_at = start_time + (intervals_passed + 1) * interval_sec


def main() -> None:  # pragma: no cover - CLI helper
    mgr = PolicyManager()

    # Full-stack mode: run macro planner loop and micro-level router in
    # parallel. The router (MA/ctdema/llm-route/router.py) is started as a
    # separate process so that it can own EllmEnv and on-node coordination,
    # while this process focuses on macro planning and evaluation.
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    # Choose a shared routing DB name and expose it so that the macro
    # planner (this process), the spawned micro router, and the spawned
    # k8s monitor all see the same <dbname>. Honour a pre-set value so the
    # E2 wrapper can inject ma-seed${N}-<ts>-style names.
    dbname_env = (os.getenv("LLM_ROUTING_DBNAME") or "").strip()
    dbname = dbname_env if dbname_env else f"llm-routing-{int(_dt.datetime.now().timestamp())}"
    os.environ["LLM_ROUTING_DBNAME"] = dbname

    # Spawn the K8s metrics monitor in the background so that
    # k8smonitor.<dbname> gets populated for EllmEnv.update_resources() to
    # read real per-node CPU/memory/GPU/network state instead of falling
    # back to the optimistic "all 1.0" defaults. Disable with
    # LLM_SPAWN_MONITOR=0 (e.g. for local diagnostics without monitor noise).
    monitor_proc = None
    if os.getenv("LLM_SPAWN_MONITOR", "1") != "0":
        monitor_path = REPO_ROOT / "monitoring.py"
        monitor_proc = subprocess.Popen(
            [sys.executable, "-u", str(monitor_path)],
            cwd=str(REPO_ROOT),
        )

    # Optionally start the micro-level router as a background process. It will
    # take care of EllmEnv, set_dbname_on_all_nodes, agent modes and per-slot
    # routing. You can disable this by setting LLM_SPAWN_MICRO=0 and running
    # MA/ctdema/llm-route/router.py manually in another terminal for easier
    # inspection of micro-level logs.
    router_proc = None
    if os.getenv("LLM_SPAWN_MICRO", "1") != "0":
        router_path = PROJECT_ROOT / "llm-route" / "router.py"
        router_proc = subprocess.Popen(
            [sys.executable, "-u", str(router_path)],
            cwd=str(REPO_ROOT),
        )

    try:
        # Run the macro planner loop in this process.
        asyncio.run(_planner_loop(mgr))
    finally:
        # Best-effort: terminate the micro router child process if we started
        # one, so that Ctrl+C on the orchestrator does not leave it behind.
        if router_proc is not None:
            try:
                router_proc.terminate()
            except Exception:
                pass
        if monitor_proc is not None:
            try:
                monitor_proc.terminate()
            except Exception:
                pass
        # Additionally, try to reset local on-node agents back to a safe
        # deterministic mode (default: 'lyapunov') so that GPT-based local agents
        # stop calling the LLM once the controller/orchestrator exits.
        shutdown_mode = os.getenv("LLM_AGENT_SHUTDOWN_MODE", "lyapunov")
        try:
            _shutdown_local_agents(mode=shutdown_mode)
        except Exception as exc:
            print(f"[orchestrator] WARNING: failed to reset local agent mode on shutdown: {exc}")


async def _ensure_agent_mode_wrapper(network: EllmEnv, mode: str = "lyapunov") -> None:
    try:
        ok_agent = await network.set_agent_mode_on_all_nodes(mode)
        if ok_agent:
            print(f"[orchestrator] Agent mode set to '{mode}'.")
        else:
            print(f"[orchestrator] WARNING: failed to set agent mode '{mode}' on some nodes.")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[orchestrator] WARNING: error setting agent mode '{mode}': {exc}")


def _shutdown_local_agents(mode: str = "lyapunov") -> None:
    """Best-effort reset of local agent mode on all worker nodes.

    This is invoked when the orchestrator process exits (including Ctrl+C)
    so that GPT-based local agents stop calling external LLMs once the
    controller is no longer driving the runtime. It does not delete any
    pods; it only switches their agent mode via /set_agent_mode.
    """
    try:
        dbname = os.getenv("LLM_ROUTING_DBNAME") or "llm-routing-shutdown"
        network = EllmEnv(dbname, REQUEST_RANGE, length_of_slot=1, status="routing")
    except Exception as exc:
        print(f"[orchestrator] WARNING: could not initialise EllmEnv for shutdown: {exc}")
        return
    try:
        asyncio.run(_ensure_agent_mode_wrapper(network, mode=mode))
    except Exception as exc:
        print(f"[orchestrator] WARNING: error during shutdown agent-mode reset: {exc}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
