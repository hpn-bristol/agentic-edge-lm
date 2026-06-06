import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

try:  # pragma: no cover - module is bundled with the worker image
    from sadrl.agent_event_logger import AgentEventLogger, get_openai_usage
except Exception:  # pragma: no cover
    try:
        from agent_event_logger import AgentEventLogger, get_openai_usage  # type: ignore
    except Exception:
        AgentEventLogger = None  # type: ignore
        def get_openai_usage(_c):  # type: ignore
            return {"tokens_in": None, "tokens_out": None}

try:  # Optional dependency – on-node agent should degrade gracefully
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # Optional dependency – K8s client for better LM placement summaries
    from kubernetes import client as k8s_client, config as k8s_config  # type: ignore
except Exception:  # pragma: no cover
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore


logger = logging.getLogger(__name__)


class _DummyMemory:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity

    def store_transition(self, state, action, reward, next_state) -> None:
        # GPT agent is stateless; we keep this for API compatibility only.
        return


def _load_node_specific_api_key() -> Optional[str]:
    """Resolve an OpenAI API key for this node.

    Priority:
    1) OPENAI_API_KEY env var (if set inside the pod).
    2) Per-node mapping from a local .env-style file (e.g. /app/sadrl/.env),
       where keys are named key1..key8 and mapped to worker1..worker8.
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    node_name = os.getenv("NODE_NAME", "")  # e.g. worker1, worker2, ...
    if not node_name:
        return None

    # Try a couple of likely paths for the on-node .env mapping file.
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        Path("/app/sadrl/.env"),
    ]
    mapping: Dict[str, str] = {}
    for path in candidates:
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k or not v:
                    continue
                mapping[k] = v
        except Exception:
            continue

    # Simple convention: worker1->key1, worker2->key2, ...
    # Fall back to key1 if nothing matches.
    idx = "".join(ch for ch in node_name if ch.isdigit())
    env_key = f"key{idx}" if idx else "key1"
    return mapping.get(env_key) or mapping.get("key1")


def _load_system_prompt(env_kind: str) -> str:
    """Load a static system prompt template for the given environment kind.

    env_kind: 'single_gpu' | 'double_gpu' | 'cpu_only'
    """
    here = Path(__file__).resolve().parent
    prompt_dir = here / "prompts"
    fname = {
        "single_gpu": "local_single_gpu.txt",
        "double_gpu": "local_double_gpu.txt",
        "cpu_only": "local_cpu_only.txt",
    }.get(env_kind, "local_single_gpu.txt")
    path = prompt_dir / fname
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        # Fallback minimal prompt if template is missing.
        return (
            "You are a local deployment controller for a single MEC node. "
            "You see queues, resource headroom and a list of valid actions. "
            "Pick exactly one valid action index that reduces latency and keeps CPU/GPU well utilised. "
            "Return a JSON object of the form {\"action\": k} with no extra text."
        )


class GPTLocalAgent:
    """
    Local GPT-based deployment agent.

    API-compatible with LyapunovAgent so that TrainingContext can use it
    without special handling.

    It observes the current local state vector and action mask, and calls a
    small GPT model to choose a single valid action index deterministically.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chkpt_dir: str,
        load_dir: str,
        lr: float,
        gamma: float,
        epsilon: float,
        batch_size: int,
        capacity: int = 1000,
        action_combo: Optional[List[List[int]]] = None,
        anchors: Optional[List[str]] = None,
        node_role: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chkpt_dir = chkpt_dir
        self.load_dir = load_dir
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.capacity = capacity
        self.action_combo: List[List[int]] = action_combo or []
        self.anchors = anchors or ["1", "2", "3", "4"]
        # High-level macro role for this node in the current epoch
        self.node_role: Dict[str, Any] = node_role or {}

        self.memory = _DummyMemory(capacity)

        # Heuristic: 4 anchors => GPU-capable env, 2 anchors => CPU-only.
        self.has_gpu = len(self.anchors) == 4
        # Try to infer env kind from vGPU-style combos if possible.
        # We only distinguish single vs double GPUs for prompt wording; the
        # actual feasibility is enforced by compute_action_mask upstream.
        self.env_kind = "cpu_only"
        if self.has_gpu:
            # Single-GPU envs have 6-bit combos like [cpu1,cpu2,g1,g2,g3,g4].
            if self.action_combo and len(self.action_combo[0]) >= 6:
                # worker5/worker8 high-capacity layouts are handled in SevEnv_double_GPUs;
                # here we just choose the generic GPU wording.
                self.env_kind = "single_gpu"
            else:
                self.env_kind = "single_gpu"

        self._base_system_prompt = _load_system_prompt(self.env_kind)
        self._system_prompt = self._base_system_prompt
        self._refresh_system_prompt_with_role()
        self._model_name = os.getenv("LOCAL_GPT_MODEL", "gpt-4o")
        self._client: Optional[Any] = None
        # Optional K8s client for summarising where LMs are running (CPU vs GPU).
        self._k8s_v1: Optional[Any] = None
        self._init_k8s_client()

        # Best-effort event logger for local LLM calls.
        try:
            self._event_logger = AgentEventLogger("local") if AgentEventLogger is not None else None
        except Exception:
            self._event_logger = None
        # Filled in by _call_local_llm so that _log_local_outcome can attach
        # latency / token usage / API error to the recorded event.
        self._last_call_meta: Dict[str, Any] = {}

    # API compatibility stubs -------------------------------------------------
    def save_checkpoint(self) -> None:
        return

    def load_checkpoint(self) -> None:
        return

    def save_replay_buffer(self, path: str = "/app/results/replay_buffer.pkl") -> None:
        return

    def load_replay_buffer(self, path: str = "/app/results/replay_buffer.pkl") -> None:
        return

    def learn(self) -> None:
        # GPT agent is stateless; no learning step required.
        return

    def _refresh_system_prompt_with_role(self) -> None:
        """Reset system prompt and, if available, append simple macro role guidance."""
        # Start from the static local template.
        self._system_prompt = self._base_system_prompt

        # Optionally append a short sentence describing which LMs this node
        # is mainly expected to handle according to the macro planner.
        try:
            role_map = self.node_role or {}
            if not isinstance(role_map, dict) or not role_map:
                return
            node_name = os.getenv("NODE_NAME", "")
            if not node_name:
                return

            raw_role = role_map.get(node_name) or {}
            if not isinstance(raw_role, dict):
                return

            # New schema: {"LMs": ["1","2",...]}
            lms = raw_role.get("LMs")
            if lms is None:
                # Backwards compatibility: cpu_LMs/gpu_LMs or cpu_anchors/gpu_anchors
                cpu_list = raw_role.get("cpu_LMs") or raw_role.get("cpu_anchors") or []
                gpu_list = raw_role.get("gpu_LMs") or raw_role.get("gpu_anchors") or []
                lms = list({str(x) for x in cpu_list} | {str(x) for x in gpu_list})

            lms = [str(x) for x in (lms or [])]
            if not lms:
                return

            lms_sorted = sorted(set(lms), key=lambda x: int(x) if x.isdigit() else x)
            lm_labels = ", ".join(f"LM{aid}" for aid in lms_sorted)
            extra = (
                "\n\n"
                f"Macro planner's command for this node is: mainly handle {lm_labels} "
                "on this node when there is work for them, and normally avoid running other LMs here."
            )
            self._system_prompt = self._base_system_prompt + extra
        except Exception:
            # Never break local scheduling if role parsing fails.
            self._system_prompt = self._base_system_prompt

    def update_node_role(self, node_role: Optional[Dict[str, Any]]) -> None:
        """Update this node's macro role and refresh the system prompt."""
        self.node_role = node_role or {}
        self._refresh_system_prompt_with_role()

    def set_dbname(self, dbname: str) -> None:
        """Tell the agent_event logger which Mongo db to write to.

        Worker pods learn their per-run dbname via /get_dbname (not env), so
        api.py forwards that value here on every (re-)initialisation.
        """
        if self._event_logger is None or not dbname:
            return
        try:
            self._event_logger.set_dbname(dbname)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    def _ensure_client(self) -> Optional[Any]:
        if OpenAI is None:
            return None
        if self._client is not None:
            return self._client
        key = _load_node_specific_api_key()
        if not key:
            return None
        os.environ["OPENAI_API_KEY"] = key
        try:
            self._client = OpenAI()
        except Exception:
            self._client = None
        return self._client

    # -------------------------------------------------------------------------
    def _init_k8s_client(self) -> None:
        """Best-effort initialisation of a Kubernetes CoreV1Api client."""
        if k8s_client is None or k8s_config is None:
            return
        if self._k8s_v1 is not None:
            return
        try:
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            self._k8s_v1 = k8s_client.CoreV1Api()
        except Exception:
            self._k8s_v1 = None

    # -------------------------------------------------------------------------
    def _maybe_log_debug(
        self,
        state_summary: Dict[str, Any],
        valid_idxs: List[int],
        user_prompt: str,
        raw_content: Optional[str],
        chosen_action: Optional[int],
    ) -> None:
        """Optionally persist the latest LLM interaction for diagnostics.

        Controlled via LOCAL_GPT_DEBUG (\"1\" to enable) and
        LOCAL_GPT_DEBUG_PATH (defaults to /app/results/gpt_local_last.json).
        """
        if os.getenv("LOCAL_GPT_DEBUG", "0") != "1":
            return
        path = os.getenv("LOCAL_GPT_DEBUG_PATH", "/app/results/gpt_local_last.json")
        payload = {
            "node_name": os.getenv("NODE_NAME", ""),
            "model": self._model_name,
            "system_prompt": self._system_prompt,
            "state_summary": state_summary,
            "valid_indices": valid_idxs,
            "user_prompt": user_prompt,
            "raw_content": raw_content,
            "chosen_action": chosen_action,
        }
        try:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Debug logging must never break local scheduling.
            pass

    def _summarise_state(self, state: Any) -> Dict[str, Any]:
        """Best-effort parse of the state vector into pod_status, backlog and resources."""
        try:
            s = np.asarray(state, dtype=float)
        except Exception:
            return {"raw": state}

        summary: Dict[str, Any] = {"raw": s.tolist()}

        if self.has_gpu:
            # Expect at least 4 (pods) + 4 (workload) + 3 (resources)
            if len(s) >= 11:
                summary["pod_status"] = s[0:4].tolist()
                summary["workload"] = s[4:8].tolist()
                summary["resources"] = s[8:11].tolist()
            else:
                summary["pod_status"] = s.tolist()
        else:
            # CPU-only env: state layout from SevEnv_no_GPU is
            # [4 pod_status, 4 workload, 3 resources] but only anchors 1/2 are actionable.
            if len(s) >= 11:
                summary["pod_status"] = s[0:4].tolist()
                summary["workload"] = s[4:8].tolist()
                summary["resources"] = s[8:11].tolist()
            else:
                summary["pod_status"] = s.tolist()
        return summary

    def _build_user_prompt(
        self,
        state_summary: Dict[str, Any],
        valid_idxs: List[int],
        action_mask,
    ) -> str:
        lines: List[str] = []
        lines.append("Here is the current local state.")
        # High-level natural language summary of resources.
        try:
            resources = state_summary.get("resources")
        except Exception:
            resources = None

        cpu_pct: Optional[float] = None
        mem_pct: Optional[float] = None
        # We derive GPU free units from the actual running GPU LMs below so
        # that it is consistent with the textual LM summary.
        node_name = os.getenv("NODE_NAME", "")
        vgpu_capacity = 0
        if self.has_gpu:
            vgpu_capacity = 4 if node_name in {"worker5", "worker8"} else 2

        if isinstance(resources, list) and len(resources) >= 3:
            cpu_free, mem_free, gpu_free = resources[:3]
            try:
                cpu_pct = max(0.0, min(100.0, float(cpu_free) * 100.0))
            except Exception:
                cpu_pct = None
            try:
                mem_pct = max(0.0, min(100.0, float(mem_free) * 100.0))
            except Exception:
                mem_pct = None

        # Summarise which LMs are running on CPU vs GPU using K8s labels
        # (llm_mode=cpu/gpu), falling back to state vector if needed.
        active_desc_done = False
        running_cpu: List[str] = []
        running_gpu: List[str] = []
        node_name = os.getenv("NODE_NAME", "")
        namespace = os.getenv("POD_NAMESPACE", None)

        if self._k8s_v1 is not None and node_name:
            try:
                if namespace:
                    pods = self._k8s_v1.list_namespaced_pod(
                        namespace=namespace,
                        field_selector=f"spec.nodeName={node_name}",
                    )
                else:
                    pods = self._k8s_v1.list_pod_for_all_namespaces(
                        field_selector=f"spec.nodeName={node_name}",
                    )
                cpu_set = set()
                gpu_set = set()
                for pod in pods.items:
                    labels = getattr(pod.metadata, "labels", None) or {}
                    llm_label = labels.get("llm")
                    if not llm_label or not llm_label.endswith(f"-{node_name}"):
                        continue
                    # llm_label is like "llm1-worker1"
                    try:
                        tail = llm_label[len("llm") :]
                        anchor = tail.split("-", 1)[0]
                    except Exception:
                        continue
                    if anchor not in {"1", "2", "3", "4"}:
                        continue
                    if pod.metadata.deletion_timestamp is not None:
                        continue
                    phase = getattr(pod.status, "phase", "")
                    if phase != "Running":
                        continue
                    mode = labels.get("llm_mode", "").lower()
                    if mode == "gpu":
                        gpu_set.add(anchor)
                    else:
                        cpu_set.add(anchor)
                running_cpu = sorted(cpu_set)
                running_gpu = sorted(gpu_set)
                if running_cpu or running_gpu:
                    parts: List[str] = []
                    if running_gpu:
                        parts.append("LM" + ", LM".join(running_gpu) + " running on GPU")
                    if running_cpu:
                        parts.append("LM" + ", LM".join(running_cpu) + " running on CPU")
                    lines.append("Right now " + "; ".join(parts) + " on this node.")
                    active_desc_done = True
            except Exception:
                # Fall back to numeric state.
                pass

        if not active_desc_done:
            try:
                pod_status = state_summary.get("pod_status")
            except Exception:
                pod_status = None
            if isinstance(pod_status, list) and len(pod_status) >= 4:
                try:
                    active_anchors = [str(i + 1) for i, score in enumerate(pod_status[:4]) if float(score) > 0.0]
                except Exception:
                    active_anchors = []
                if active_anchors:
                    if self.has_gpu:
                        lines.append(
                            "Right now LM" + ", LM".join(active_anchors) + " appear to be active (CPU/GPU mixed) on this node."
                        )
                    else:
                        lines.append(
                            "Right now LM" + ", LM".join(active_anchors) + " are running on CPU on this node."
                        )
                    active_desc_done = True

        if not active_desc_done:
            lines.append("Right now no LMs appear to be running on this node.")

        # After we know which LMs are on GPU, derive an approximate GPU
        # free count so that it stays consistent with the LM summary.
        gpu_units_free: Optional[float] = None
        if self.has_gpu and vgpu_capacity > 0:
            try:
                used_gpu_units = float(len(running_gpu))
                gpu_units_free = max(0.0, float(vgpu_capacity) - used_gpu_units)
            except Exception:
                gpu_units_free = None

        # Finally emit the resource summary line (CPU/MEM from metrics,
        # GPU from running LMs).
        res_parts: List[str] = []
        if cpu_pct is not None:
            res_parts.append(f"approximately {cpu_pct:.0f}% CPU headroom")
        if mem_pct is not None:
            res_parts.append(f"{mem_pct:.0f}% memory headroom")
        if gpu_units_free is not None:
            res_parts.append(f"{gpu_units_free:.1f} / {vgpu_capacity} vGPU units free")
        if res_parts:
            # Prepend this ahead of the LM activity sentence so that both
            # refer to the same notion of GPU usage.
            lines.insert(1, "This node now has " + ", ".join(res_parts) + ".")

        # Natural language summary of per-LM queues (workload).
        try:
            workload = state_summary.get("workload")
        except Exception:
            workload = None
        if isinstance(workload, list) and len(workload) >= 4:
            try:
                w_vals = [max(0, int(float(x))) for x in workload[:4]]
            except Exception:
                w_vals = []
            if w_vals:
                labels = ["LM1", "LM2", "LM3", "LM4"]
                parts: List[str] = []
                for label, w in zip(labels, w_vals):
                    parts.append(f"{label} has {w} queued requests")
                lines.append("Current queues: " + ", ".join(parts) + ".")

        return "\n".join(lines)

    def _filter_valid_indices_by_role(self, valid_idxs: List[int]) -> List[int]:
        """No-op hook: macro role is advisory only, not a hard filter."""
        return valid_idxs

    def _decode_combo_sets(self, combo: List[int]) -> Dict[str, set]:
        """Decode an action combo into cpu_on/gpu_on anchor sets."""
        cpu_on: set = set()
        gpu_on: set = set()
        if self.has_gpu:
            # [cpu1, cpu2, gpu1, gpu2, gpu3, gpu4]
            cpu1 = combo[0] if len(combo) > 0 else 0
            cpu2 = combo[1] if len(combo) > 1 else 0
            gpu1 = combo[2] if len(combo) > 2 else 0
            gpu2 = combo[3] if len(combo) > 3 else 0
            gpu3 = combo[4] if len(combo) > 4 else 0
            gpu4 = combo[5] if len(combo) > 5 else 0
            if cpu1 == 1:
                cpu_on.add("1")
            if cpu2 == 1:
                cpu_on.add("2")
            if gpu1 == 1:
                gpu_on.add("1")
            if gpu2 == 1:
                gpu_on.add("2")
            if gpu3 == 1:
                gpu_on.add("3")
            if gpu4 == 1:
                gpu_on.add("4")
        else:
            # CPU-only: [cpu1, cpu2]
            cpu1 = combo[0] if len(combo) > 0 else 0
            cpu2 = combo[1] if len(combo) > 1 else 0
            if cpu1 == 1:
                cpu_on.add("1")
            if cpu2 == 1:
                cpu_on.add("2")
        return {"cpu_on": cpu_on, "gpu_on": gpu_on}

    def _choose_nearest_action(
        self,
        desired_cpu_on: set,
        desired_gpu_on: set,
        valid_idxs: List[int],
    ) -> int:
        """Map desired cpu_on/gpu_on sets to the nearest feasible action index."""
        best_idx = None
        best_dist = 1e9
        for idx in valid_idxs:
            if idx < 0 or not self.action_combo or idx >= len(self.action_combo):
                continue
            combo = self.action_combo[idx]
            if any(v < 0 for v in combo):
                # KEEP: consider but only if nothing else is close
                sets = self._decode_combo_sets(combo)
            else:
                sets = self._decode_combo_sets(combo)
            cpu_on_c = sets["cpu_on"]
            gpu_on_c = sets["gpu_on"]
            d = 0
            d += len(desired_cpu_on.symmetric_difference(cpu_on_c))
            d += len(desired_gpu_on.symmetric_difference(gpu_on_c))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            # Fallback: random valid
            return int(random.choice(valid_idxs))
        return int(best_idx)

    def _parse_text_plan(self, content: str) -> Tuple[set, set]:
        """Parse natural-language plan like 'place LM1 on GPU, LM2 on CPU, LM3 off, LM4 off'."""
        text = content.strip()
        if not text:
            raise ValueError("empty content")

        desired_cpu_on: set = set()
        desired_gpu_on: set = set()
        anchors_seen: set = set()

        # Split into clauses by commas; fall back to whole text if no comma.
        clauses = [c.strip() for c in text.split(",") if c.strip()] or [text]

        for clause in clauses:
            lower = clause.lower()
            for a in self.anchors:
                token = f"lm{a}"
                if token.lower() in lower:
                    anchors_seen.add(a)
                    if "gpu" in lower:
                        desired_gpu_on.add(a)
                        desired_cpu_on.discard(a)
                    elif "cpu" in lower:
                        desired_cpu_on.add(a)
                        desired_gpu_on.discard(a)
                    elif "off" in lower:
                        desired_cpu_on.discard(a)
                        desired_gpu_on.discard(a)
                    # If none of gpu/cpu/off is mentioned, leave this anchor as-is (treated as off by default).

        if not anchors_seen:
            raise ValueError("no LM anchors found in text plan")

        # CPU-only env: ignore any GPU assignments.
        if not self.has_gpu:
            desired_gpu_on = set()

        return desired_cpu_on, desired_gpu_on

    def _log_local_outcome(
        self,
        outcome: str,
        *,
        raw: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
    ) -> None:
        """Forward one event to the agent_event logger, if available."""
        if self._event_logger is None:
            return
        # Refresh dbname on the fly in case the controller set it via env later.
        try:
            db = os.getenv("LLM_ROUTING_DBNAME") or ""
            if db:
                self._event_logger.set_dbname(db)
        except Exception:
            pass
        meta = self._last_call_meta or {}
        try:
            self._event_logger.log(
                outcome,
                node=os.getenv("NODE_NAME", ""),
                model=self._model_name,
                tokens_in=meta.get("tokens_in"),
                tokens_out=meta.get("tokens_out"),
                latency_ms=meta.get("latency_ms"),
                error=meta.get("error"),
                raw_truncated=raw,
                validation_errors=validation_errors,
            )
        except Exception:
            pass

    def choose_action(self, state, action_mask) -> int:
        """Call the local GPT policy to pick a single valid action index."""
        mask_arr = np.asarray(action_mask, dtype=bool)
        base_valid_idxs = np.nonzero(mask_arr)[0].tolist()
        if not base_valid_idxs:
            return random.randrange(self.action_dim)

        # Apply hard macro-role constraints at the action level so that this
        # node cannot enable forbidden anchors in normal operation.
        valid_idxs = self._filter_valid_indices_by_role(base_valid_idxs)
        state_summary = self._summarise_state(state)
        client = self._ensure_client()
        if client is None:
            # Fallback: simple random valid action if GPT client is not available.
            logger.warning(
                "[GPT-LOCAL] OpenAI client is not available on node=%s; falling back to random valid action.",
                os.getenv("NODE_NAME", ""),
            )
            self._last_call_meta = {"error": "no_openai_client"}
            self._log_local_outcome("api_error")
            self._log_local_outcome("fallback_random")
            return int(random.choice(valid_idxs))

        user_prompt = self._build_user_prompt(state_summary, valid_idxs, action_mask)

        self._last_call_meta = {}
        t0 = time.monotonic()
        try:
            logger.info(
                "[GPT-LOCAL] Calling local GPT model='%s' with %d candidate actions on node=%s",
                self._model_name,
                len(valid_idxs),
                os.getenv("NODE_NAME", ""),
            )
            completion = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content if completion.choices else None
            usage = get_openai_usage(completion)
            self._last_call_meta = {
                "latency_ms": int((time.monotonic() - t0) * 1000),
                "tokens_in": usage.get("tokens_in"),
                "tokens_out": usage.get("tokens_out"),
            }
        except Exception as e:
            # On any failure, fall back to a random valid action.
            logger.error(
                "[GPT-LOCAL] Error during GPT call on node=%s: %s",
                os.getenv("NODE_NAME", ""),
                e,
                exc_info=True,
            )
            self._last_call_meta = {
                "latency_ms": int((time.monotonic() - t0) * 1000),
                "error": str(e),
            }
            self._log_local_outcome("api_error")
            self._log_local_outcome("fallback_random")
            self._maybe_log_debug(state_summary, valid_idxs, user_prompt, None, None)
            return int(random.choice(valid_idxs))

        if not content:
            logger.warning(
                "[GPT-LOCAL] GPT model='%s' returned empty content on node=%s; using fallback action.",
                self._model_name,
                os.getenv("NODE_NAME", ""),
            )
            self._log_local_outcome("parse_fail", raw="<empty>")
            self._log_local_outcome("fallback_random")
            self._maybe_log_debug(state_summary, valid_idxs, user_prompt, None, None)
            return int(random.choice(valid_idxs))

        # First try to interpret the response as JSON (backward compatibility).
        try:
            text = content.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                obj = json.loads(text[start : end + 1])

                # New-style JSON API: cpu_on/gpu_on sets.
                if "cpu_on" in obj or "gpu_on" in obj:
                    desired_cpu_on = set(str(a) for a in obj.get("cpu_on", []))
                    desired_gpu_on = set(str(a) for a in obj.get("gpu_on", []))
                    chosen_idx = self._choose_nearest_action(desired_cpu_on, desired_gpu_on, valid_idxs)
                    self._log_local_outcome("success")
                    self._maybe_log_debug(state_summary, valid_idxs, user_prompt, content, chosen_idx)
                    logger.info(
                        "[GPT-LOCAL] GPT model='%s' requested cpu_on=%s gpu_on=%s; mapped to action=%d on node=%s",
                        self._model_name,
                        sorted(desired_cpu_on),
                        sorted(desired_gpu_on),
                        chosen_idx,
                        os.getenv("NODE_NAME", ""),
                    )
                    return int(chosen_idx)

                # Alternative JSON API: direct action index.
                if "action" in obj:
                    k = int(obj.get("action"))
                    if k in valid_idxs:
                        self._log_local_outcome("success")
                        self._maybe_log_debug(state_summary, valid_idxs, user_prompt, content, k)
                        logger.info(
                            "[GPT-LOCAL] GPT model='%s' chose valid indexed action=%d on node=%s",
                            self._model_name,
                            k,
                            os.getenv("NODE_NAME", ""),
                        )
                        return k
        except Exception:
            logger.warning(
                "[GPT-LOCAL] Failed to parse GPT response as JSON on node=%s; will try to interpret natural-language plan.",
                os.getenv("NODE_NAME", ""),
            )

        # Then try to interpret the response as a natural-language plan like
        # "place LM1 on GPU, LM2 on CPU, LM3 off, LM4 off".
        try:
            desired_cpu_on, desired_gpu_on = self._parse_text_plan(content)
            chosen_idx = self._choose_nearest_action(desired_cpu_on, desired_gpu_on, valid_idxs)
            self._log_local_outcome("success")
            self._maybe_log_debug(state_summary, valid_idxs, user_prompt, content, chosen_idx)
            logger.info(
                "[GPT-LOCAL] GPT model='%s' text-plan cpu_on=%s gpu_on=%s; mapped to action=%d on node=%s",
                self._model_name,
                sorted(desired_cpu_on),
                sorted(desired_gpu_on),
                chosen_idx,
                os.getenv("NODE_NAME", ""),
            )
            return int(chosen_idx)
        except Exception:
            logger.warning(
                "[GPT-LOCAL] Failed to interpret GPT response as a placement plan on node=%s; falling back to random action.",
                os.getenv("NODE_NAME", ""),
            )
            self._log_local_outcome("parse_fail", raw=content)
            self._maybe_log_debug(state_summary, valid_idxs, user_prompt, content, None)

        # Last resort: random valid action.
        fallback = int(random.choice(valid_idxs))
        self._log_local_outcome("fallback_random")
        self._maybe_log_debug(state_summary, valid_idxs, user_prompt, content, fallback)
        logger.info(
            "[GPT-LOCAL] Using fallback random action=%d on node=%s",
            fallback,
            os.getenv("NODE_NAME", ""),
        )
        return fallback
