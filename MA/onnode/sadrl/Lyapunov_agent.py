import json
import os
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class _DummyMemory:
    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def __len__(self):
        return len(self.buffer)


class LyapunovAgent:
    """
    Drift-plus-penalty style local controller with the same external API as the local deployment agents.

    Score(a) = sum_k Q_k * mu_hat_k(a)  -  V * ( GPU_shadow_price(a) + churn_penalty(a) )

    - Q_k: backlog per anchor from state (workload segment)
    - mu_hat_k(a): proxy service headroom using f(cpu_free), f(mem_free), f(gpu_free)
    - GPU_shadow_price(a) = p0 + p1*(1-gpu_free) + p2*img_share, multiplied by number of GPU instances turned on in a
    - churn_penalty(a) = lambda * sum_k |intent_k(a) - running_k| * (1 + kappa * Q_k)

    Capacity/feasibility is enforced upstream by action_mask.
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
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = _DummyMemory(capacity)
        self.action_combo = action_combo or []
        self.anchors = anchors or ["1", "2", "3", "4"]

        # DPP weight
        self.V = float(os.getenv("DPP_V", "1.0"))

        # Type weights (per-LM, per-mode). Literal defaults are kept here so
        # that apply_hyper.py's regex rewrite can persist calibrated alphas.
        # Image LMs (llm3/llm4) cannot run on CPU, so their CPU weights are 0.
        if len(self.anchors) == 4:
            llm1_cpu = float(os.getenv("LLM1_CPU_PREF", "3.678337201089698"))
            llm2_cpu = float(os.getenv("LLM2_CPU_PREF", "1.0"))
            self.alpha_cpu = [llm1_cpu, llm2_cpu, 0.0, 0.0]
            llm1_gpu = float(os.getenv("LLM1_GPU_PREF", "5.802139037433155"))
            llm2_gpu = float(os.getenv("LLM2_GPU_PREF", "7.198278743315509"))
            llm3_gpu = float(os.getenv("LLM3_GPU_PREF", "5.611769282995263"))
            llm4_gpu = float(os.getenv("LLM4_GPU_PREF", "3.024431376175736"))
            self.alpha_gpu = [llm1_gpu, llm2_gpu, llm3_gpu, llm4_gpu]
        else:
            self.alpha_cpu = [1.0, 1.0]
            self.alpha_gpu = [0.0, 0.0]

        # Churn / price parameters. Keep the env names and literal defaults
        # stable so calibration scripts can patch them mechanically.
        self.lambda_churn = float(os.getenv("CHURN_LAMBDA", "0.1"))
        self.kappa_workload = float(os.getenv("CHURN_KAPPA", "1.0"))

        self.price_p0 = float(os.getenv("GPU_PRICE_P0", "0.0"))
        self.price_p1 = float(os.getenv("GPU_PRICE_P1", "0.15"))
        self.price_p2 = float(os.getenv("GPU_PRICE_P2", "0.3"))

        self.text_gpu_speedup = float(os.getenv("TEXT_GPU_SPEEDUP", "1.0"))
        self.tiny_noise = float(os.getenv("LYAPUNOV_TINY_NOISE", "1e-6"))

        self.priority = self._load_priority()
        self.coverage_bonus = float(os.getenv("LYAP_COVERAGE_BONUS", "0.0"))
        self.coverage_threshold = float(os.getenv("LYAP_COVERAGE_THRESHOLD", "0.0"))

    # API compatibility
    def save_checkpoint(self):
        return

    def load_checkpoint(self):
        return

    def save_replay_buffer(self, path: str = "/app/results/replay_buffer.pkl"):
        return

    def load_replay_buffer(self, path: str = "/app/results/replay_buffer.pkl"):
        return

    def learn(self):
        return

    def _parse_state(self, state):
        s = np.asarray(state, dtype=np.float32)

        if len(self.anchors) == 4:
            # [4 pod, 4 backlog, 3 resources]
            if len(s) < 11:
                # fallback with zeros
                pod_status = np.zeros(4, dtype=np.float32)
                backlog = np.zeros(4, dtype=np.float32)
                resources = np.zeros(3, dtype=np.float32)
            else:
                pod_status = s[0:4]
                backlog = s[4:8]
                resources = s[8:11]
            return backlog, pod_status, resources, True
        else:
            if len(s) < 6:
                pod_status = np.zeros(2, dtype=np.float32)
                backlog = np.zeros(2, dtype=np.float32)
                resources = np.zeros(2, dtype=np.float32)
            else:
                pod_status = s[0:2]
                backlog = s[2:4]
                resources = s[4:6]
            return backlog, pod_status, resources, False

    def _load_priority(self) -> Dict[str, float]:
        raw = os.getenv("LYAP_PRIORITY_WEIGHTS")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            return {}

    def choose_action(self, state, action_mask) -> int:
        mask_arr = np.asarray(action_mask, dtype=bool)
        valid_idxs = np.nonzero(mask_arr)[0].tolist()
        if not valid_idxs:
            return random.randrange(self.action_dim)

        Q, running, res, has_gpu = self._parse_state(state)
        # resources
        if has_gpu:
            cpu_free = float(res[0]) if len(res) > 0 else 0.5
            mem_free = float(res[1]) if len(res) > 1 else 0.5
            gpu_free = float(res[2]) if len(res) > 2 else 0.0
        else:
            cpu_free = float(res[0]) if len(res) > 0 else 0.5
            mem_free = float(res[1]) if len(res) > 1 else 0.5
            gpu_free = 0.0

        # image share for shadow price
        if len(Q) >= 4:
            total_Q = float(np.sum(Q))
            img_share = float((Q[2] + Q[3]) / total_Q) if total_Q > 1e-6 else 0.0
        else:
            img_share = 0.0

        price_gpu = self.price_p0 + self.price_p1 * (1.0 - gpu_free) + self.price_p2 * img_share

        def f(x: float) -> float:
            return 0.6 + 0.4 * float(x)

        def score_combo_4(combo: List[int]) -> float:
            cpu1, cpu2, g1, g2, g3, g4 = combo
            # ---------- drift term: how much this action helps drain queues ----------
            drift_term = 0.0
            # LLM1
            if cpu1 == 1:
                drift_term += Q[0] * self.alpha_cpu[0] * f(cpu_free) * f(mem_free)
            if g1 == 1:
                drift_term += Q[0] * (self.alpha_gpu[0] * self.text_gpu_speedup) * f(gpu_free)
            # LLM2
            if cpu2 == 1:
                drift_term += Q[1] * self.alpha_cpu[1] * f(cpu_free) * f(mem_free)
            if g2 == 1:
                drift_term += Q[1] * (self.alpha_gpu[1] * self.text_gpu_speedup) * f(gpu_free)
            # LLM3/4 (GPU only)
            if g3 == 1:
                drift_term += Q[2] * self.alpha_gpu[2] * f(gpu_free)
            if g4 == 1:
                drift_term += Q[3] * self.alpha_gpu[3] * f(gpu_free)

            # ---------- penalty components: churn + GPU price ----------
            intent = [max(cpu1, g1), max(cpu2, g2), g3, g4]
            churn = 0.0
            for i in range(4):
                desired = float(intent[i])
                run = 1.0 if running[i] >= 0.9 else 0.0
                churn += abs(desired - run) * (1.0 + self.kappa_workload * Q[i])
            churn *= self.lambda_churn

            # GPU shadow price * number of GPU instances
            gpu_used = float(g1 + g2 + g3 + g4)
            price = price_gpu * gpu_used

            # optional priority bonus (LLM-specific bias from central router)
            if self.priority:
                for anchor, active in zip(["1", "2", "3", "4"], [max(cpu1, g1), max(cpu2, g2), g3, g4]):
                    if active:
                        drift_term += float(self.priority.get(anchor, 0.0))

            # optional coverage penalty when backlog is high but anchor kept inactive
            if self.coverage_bonus > 0.0 and self.coverage_threshold > 0.0:
                coverage_pen = 0.0
                for idx, active in enumerate([max(cpu1, g1), max(cpu2, g2), g3, g4]):
                    if active:
                        continue
                    if Q[idx] >= self.coverage_threshold:
                        coverage_pen += self.coverage_bonus * Q[idx]
                churn += coverage_pen

            penalty_term = price + churn
            # DPP-style score: drift_term - V * penalty_term
            return float(drift_term - self.V * penalty_term)

        def score_combo_2(combo: List[int]) -> float:
            c1, c2 = combo
            # ---------- drift term on CPU-only env ----------
            drift_term = 0.0
            if c1 == 1:
                drift_term += Q[0] * self.alpha_cpu[0] * f(cpu_free) * f(mem_free)
            if c2 == 1:
                drift_term += Q[1] * self.alpha_cpu[1] * f(cpu_free) * f(mem_free)

            intent = [c1, c2]
            churn = 0.0
            for i in range(2):
                desired = float(intent[i])
                run = 1.0 if running[i] >= 0.9 else 0.0
                churn += abs(desired - run) * (1.0 + self.kappa_workload * Q[i])
            churn *= self.lambda_churn

            if self.priority:
                for anchor, active in zip(self.anchors, intent):
                    if active:
                        drift_term += float(self.priority.get(anchor, 0.0))

            if self.coverage_bonus > 0.0 and self.coverage_threshold > 0.0:
                for idx, active in enumerate(intent):
                    if not active and Q[idx] >= self.coverage_threshold:
                        churn += self.coverage_bonus * Q[idx]

            # no GPU price on CPU-only env
            penalty_term = churn
            return float(drift_term - self.V * penalty_term)

        best_idx = None
        best_score = -1e9
        for idx in valid_idxs:
            if idx < 0 or idx >= self.action_dim:
                continue
            if not self.action_combo or idx >= len(self.action_combo):
                score = 0.0
            else:
                combo = self.action_combo[idx]
                # explicit KEEP sentinel (any value not in {0,1})
                if any((v not in (0, 1)) for v in combo):
                    score = -1e-6  # prefer real changes over KEEP when possible
                else:
                    score = score_combo_4(combo) if len(self.anchors) == 4 else score_combo_2(combo)

            score += self.tiny_noise * random.random()
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return random.choice(valid_idxs)
        return int(best_idx)
