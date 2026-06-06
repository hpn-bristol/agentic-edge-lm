"""Online weights correction heuristics for the router."""

from __future__ import annotations

from typing import Any, Dict


def _safe_norm(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def apply_online_correction(
    base_policy: Dict[str, Any],
    recent_metrics: Dict[str, Any],
    beta: float = 0.7,
) -> Dict[str, Any]:
    """Blend policy weights with short-window latency signals.

    Parameters
    ----------
    base_policy:
        Policy dictionary containing ``routes``.
    recent_metrics:
        Snapshot subset with ``anchors`` → node → metrics.
    beta:
        Interpolation factor between static policy (beta) and feedback (1-beta).
    """

    beta = min(1.0, max(0.0, beta))
    corrected = {"routes": {}, "meta": dict(base_policy.get("meta", {}))}

    anchors_metrics = (recent_metrics or {}).get("anchors", {})
    for anchor, weights in base_policy.get("routes", {}).items():
        feedback = anchors_metrics.get(anchor, {})
        blended = {}
        for node, weight in weights.items():
            fb = feedback.get(node, {})
            # Reward nodes with lower short-window mean latency.
            # We treat 900s as capped latency values inside mean_latency;
            # skip / missing samples are ignored upstream when computing means.
            mean_latency = float(fb.get("mean_latency", 0.0))
            if mean_latency <= 0.0:
                # No reliable latency yet: fall back to neutral score.
                score = 1.0
            else:
                score = 1.0 / mean_latency
            blended[node] = beta * float(weight) + (1.0 - beta) * score
        corrected["routes"][anchor] = _safe_norm(blended)

    return corrected
