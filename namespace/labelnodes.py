from kubernetes import client, config


config.load_kube_config()
v1 = client.CoreV1Api()


GPU_LABELS = {
    "dedicated": "worker",
    "feature.node.kubernetes.io/pci-10de.present": "true",
    "nvidia.com/gpu.present": "true",
    "nvidia.com/mps.capable": "true",
}


NODE_LABELS = {
    "worker1": {
        "workerid": "worker1",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "gpuworker",
    },
    "worker2": {
        "workerid": "worker2",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "gpuworker",
    },
    "worker3": {
        "workerid": "worker3",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "gpuworker",
    },
    "worker4": {
        "workerid": "worker4",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "gpuworker",
    },
    "worker5": {
        "workerid": "worker5",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "biggpuworker",
    },
    "worker6": {
        "workerid": "worker6",
        "dedicated": "worker",
    },
    "worker7": {
        "workerid": "worker7",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "gpuworker",
    },
    "worker8": {
        "workerid": "worker8",
        **GPU_LABELS,
        "nvidia.com/device-plugin.config": "biggpuworker",
    },
    "controller": {
        "workerid": "controller",
        "dedicated": "controller",
    },
}


for node_name, labels in NODE_LABELS.items():
    node = v1.read_node(node_name)
    current_labels = node.metadata.labels or {}
    current_labels.update(labels)
    v1.patch_node(node_name, {"metadata": {"labels": current_labels}})
    print(f"Node {node_name} labeled with {labels}")


# This file labels nodes so that Kubernetes selectors and the NVIDIA device
# plugin can map workload pods to the intended worker group.
