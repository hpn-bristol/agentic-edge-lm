"""Cluster static information helpers for LLM routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from kubernetes import client, config


@dataclass
class NodeStaticInfo:
    name: str
    ip: str
    cpu_cores: Optional[float]
    memory_gib: Optional[float]
    gpu_replicas: int


MODEL_REQUIREMENTS = {
    "1": {"description": "Text LLM (small)", "ram_gib": 3.0, "vram_gib": 0.0, "gpu_required": False},
    "2": {"description": "Text LLM (large)", "ram_gib": 9.0, "vram_gib": 0.0, "gpu_required": False},
    "3": {"description": "Image generation", "ram_gib": 2.5, "vram_gib": 1.0, "gpu_required": True},
    "4": {"description": "Image understanding", "ram_gib": 3.5, "vram_gib": 1.0, "gpu_required": True},
}


def _to_gib(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip().upper()
    try:
        if value.endswith("KI"):
            return float(value[:-2]) / (1024 ** 2)
        if value.endswith("MI"):
            return float(value[:-2]) / 1024
        if value.endswith("GI"):
            return float(value[:-2])
        if value.endswith("TI"):
            return float(value[:-2]) * 1024
        return float(value) / (1024 ** 3)
    except Exception:
        return None


def _to_cpu(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        try:
            if value.endswith("m"):
                return float(value[:-1]) / 1000.0
        except Exception:
            pass
    return None


def fetch_cluster_static_info() -> Dict[str, NodeStaticInfo]:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    nodes = {}
    # Logical GPU replica overrides for what the LLM should see in the
    # static context. This does not affect actual scheduling; Lyapunov
    # and Kubernetes continue to use allocatable nvidia.com/gpu.
    override_gpu_replicas = {
        "worker1": 2,
        "worker2": 2,
        "worker3": 2,
        "worker4": 2,
        "worker5": 4,
        "worker6": 0,
        "worker7": 2,
        "worker8": 4,
    }
    for node in v1.list_node().items:
        labels = node.metadata.labels or {}
        if labels.get("node-role.kubernetes.io/controlplane"):
            continue
        hostname = node.metadata.name
        ips = [addr.address for addr in (node.status.addresses or []) if addr.type == "InternalIP" and addr.address]
        if not ips:
            continue
        alloc = node.status.allocatable or {}
        cpu = _to_cpu(alloc.get("cpu"))
        mem = _to_gib(alloc.get("memory"))
        replicas_label = labels.get("nvidia.com/gpu.replicas")
        gpu_replicas = 0
        if replicas_label and str(replicas_label).isdigit():
            gpu_replicas = int(replicas_label)
        elif alloc.get("nvidia.com/gpu") and str(alloc["nvidia.com/gpu"]) != "0":
            try:
                gpu_replicas = int(alloc["nvidia.com/gpu"])
            except Exception:
                gpu_replicas = 0
        # Apply logical overrides for LLM-facing static context.
        if hostname in override_gpu_replicas:
            gpu_replicas = override_gpu_replicas[hostname]
        nodes[ips[0]] = NodeStaticInfo(
            name=hostname,
            ip=ips[0],
            cpu_cores=cpu,
            memory_gib=mem,
            gpu_replicas=gpu_replicas,
        )
    return nodes


def build_static_context_text() -> str:
    """Return a high-level, LLM-friendly description of the cluster layout.

    We deliberately hide low-level CPU/memory numbers and instead present
    an abstract GPU-slot view that matches how macro/micro planners reason
    about capacity and fairness.
    """
    nodes = fetch_cluster_static_info()
    lines = [
        "Cluster configuration:",
        "Use the node IPs listed below as the keys in LM_policies.",
    ]
    if nodes:
        for info in sorted(nodes.values(), key=lambda x: x.name):
            if info.gpu_replicas <= 0:
                cap_text = "has no GPU, and ideally should not receive GPU-bound tasks"
            elif info.gpu_replicas >= 4:
                cap_text = (
                    f"has high GPU capacity ({info.gpu_replicas} logical GPU units), "
                    "and can run up to three LMs simultaneously in total"
                )
            else:
                cap_text = (
                    f"has GPU capacity ({info.gpu_replicas} logical GPU units), "
                    "and can run up to two LMs simultaneously"
                )
            lines.append(f"Node {info.name} ({info.ip}): {cap_text}.")
    else:
        lines.append(
            "No worker nodes were discovered from Kubernetes; check node labels and cluster access."
        )
    lines.append(
        "To achieve your objective, ensure that GPUs are effectively utilized and avoid sending more LM types to a server than it can run concurrently."
    )
    return "\n".join(lines)
