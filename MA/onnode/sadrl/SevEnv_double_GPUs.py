import re
import os
import tarfile
from datetime import datetime
import copy
import math
import random
import numpy as np
from PIL import Image
from kubernetes import client, config, watch
from kubernetes.client import V1DeleteOptions
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
import json
import time
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
import logging

logger = logging.getLogger(__name__)

class SevEnv_double_GPUs():
    def __init__(self, dbname, length_of_slot, cooldown_period):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.client = client

        self.v1 = self.client.CoreV1Api()

        service = self.v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        self.mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")
        
        self.api_instance = self.client.AppsV1Api()
        self.resource_col = self.mongo["k8smonitor"][dbname]

        self.ObjectApi = self.client.CustomObjectsApi()

        self.memory_require = ["3Gi", "9Gi", "2.5Gi", "3.5Gi"]

        self.pod_name = os.getenv('HOSTNAME')
        self.namespace = os.getenv("POD_NAMESPACE")
        self.node_name = os.getenv("NODE_NAME")
        self.pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={self.node_name}')
        self.node_ip = [x.address for x in self.v1.read_node(name=self.node_name).status.addresses if x.type == 'InternalIP'][0]
        self.prompt_col = self.mongo["prompt_recorder"][f"{self.node_ip.replace('.', '_')}"]
        self.latency_col = self.mongo["latency"][f"{self.node_ip.replace('.', '_')}"]
        self.reward_ma_col = self.mongo["reward"]["ma_reward"]
        self.reward_sa_col = self.mongo["reward"]["sa_reward"]
        
        self.node = self.v1.read_node(name=self.node_name)
        self.cpu_upperlimit = self.node.status.capacity.get("cpu")

        # self.local_slot_anchor = 0
        self.llm_amount = 4
        self.length = length_of_slot
        self.vGPU = 4
        self.state = None

        self.cooldown_period = cooldown_period
        self.cooldown_info = {}
        for anchor in range(1, 5):
            llm_key = f"llm{anchor}"
            self.cooldown_info[llm_key] = {
                "remaining_steps": 0,
                "last_action": None,
            }

        # Action layout: [cpu1, cpu2, gpu1, gpu2, gpu3, gpu4]
        # On double-GPU nodes (vGPU=4), construct the action space so that:
        #   - LM1/LM2 can be off / CPU / GPU;
        #   - LM3/LM4 can be off / GPU (no CPU);
        #   - at most 3 LMs are on GPU concurrently;
        #   - at most 1 LM is on CPU concurrently.
        # This ensures that for any high-level plan that respects these
        # constraints, there is an exact matching action.
        self.action_combo = []
        statuses12 = ["off", "cpu", "gpu"]
        statuses34 = ["off", "gpu"]
        seen: set[tuple[int, int, int, int, int, int]] = set()
        for s1 in statuses12:
            for s2 in statuses12:
                for s3 in statuses34:
                    for s4 in statuses34:
                        cpu_count = int(s1 == "cpu") + int(s2 == "cpu")
                        gpu_count = (
                            int(s1 == "gpu")
                            + int(s2 == "gpu")
                            + int(s3 == "gpu")
                            + int(s4 == "gpu")
                        )
                        if cpu_count > 1:
                            continue
                        if gpu_count > 3:
                            continue
                        combo = [0, 0, 0, 0, 0, 0]
                        # LM1
                        if s1 == "cpu":
                            combo[0] = 1
                        elif s1 == "gpu":
                            combo[2] = 1
                        # LM2
                        if s2 == "cpu":
                            combo[1] = 1
                        elif s2 == "gpu":
                            combo[3] = 1
                        # LM3
                        if s3 == "gpu":
                            combo[4] = 1
                        # LM4
                        if s4 == "gpu":
                            combo[5] = 1
                        key = tuple(combo)
                        if key not in seen:
                            seen.add(key)
                            self.action_combo.append(combo)
        # explicit KEEP action (sentinel)
        self.action_combo.append([-1, -1, -1, -1, -1, -1])

        self.current_deploy = [0,0,0,0,0,0]
        self.deploy_plan = [0,0,0,0,0,0]
        self._anchor_stats = None

    async def is_pod_pending(self, llm_anchor: str) -> bool:
        pod_label = f"llm=llm{llm_anchor}-{self.node_name}"
        pods = self.v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=pod_label
        )
        for pod in pods.items:
            if pod.status.phase == "Pending":
                return True
        return False
    
    async def is_pod_busy(self, llm_anchor: int):
        pod_name = f"llm{llm_anchor}-{self.node_name}"
        pods = self.v1.list_pod_for_all_namespaces(field_selector=f"metadata.name={pod_name},spec.nodeName={self.node_name}").items

        if not pods:
            return False
        pod = pods[0]
        if pod.status.phase == "Pending":
            return True
        if pod.metadata.deletion_timestamp is not None:
            return True
        return False
    
    async def update_resources_async(self):
        data = self.resource_col.find().sort("timestamp", -1).limit(1)
        docs = await data.to_list(length=1)
        
        if not docs:
            cpu = 1.0
            memory = 1.0
            gpu = 1.0
            self.total_memory_mb = None
            self.free_memory_mb = None
        else:
            doc = docs[0]
            metrics = doc['metrics']
            cpu = 1 - metrics[self.node_name]['cpu']/100
            try:
                total_mb = float(metrics[self.node_name]['total_memory'])
                free_mb = float(metrics[self.node_name]['free_memory'])/1024.0
                memory = free_mb / total_mb if total_mb > 0 else 1.0
                self.total_memory_mb = total_mb
                self.free_memory_mb = free_mb
            except Exception:
                memory = 1.0
                self.total_memory_mb = None
                self.free_memory_mb = None
            # prefer ConfigMap 'remaining-gpu' as authoritative remaining units
            gpu = None
            try:
                cm = self.v1.read_namespaced_config_map(name="remaining-gpu", namespace=self.namespace)
                data_dict = cm.data or {}
                val = data_dict.get(self.node_name)
                if val is not None:
                    rem_units = max(0, int(val))
                    gpu = float(rem_units) / float(self.vGPU) if self.vGPU > 0 else 0.0
            except Exception:
                gpu = None
            if gpu is None:
                # fallback to ratio from Node allocatable
                try:
                    frac = await asyncio.to_thread(self.get_remaining_gpus, self.node_name)
                    gpu = float(frac) if frac is not None else 0.0
                except Exception:
                    gpu = 0.0
            
        self.current_resource = [cpu, memory, gpu]

    def get_remaining_gpus(self, node_name):
        node = self.v1.read_node(node_name)
        allocatable_gpus = node.status.allocatable.get('nvidia.com/gpu')
        if allocatable_gpus:
            allocatable_gpus = int(allocatable_gpus)
        else:
            print(f"Node {node_name} does not have GPUs or the nvidia.com/gpu resource is not found.")
            return None
        requested_gpus = 0
        pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node_name}')
        for pod in pods.items:
            for container in pod.spec.containers:
                if container.resources and container.resources.requests:
                    gpu_request = container.resources.requests.get('nvidia.com/gpu')
                    if gpu_request:
                        requested_gpus += int(gpu_request)
        remaining_gpus = (allocatable_gpus - requested_gpus) / allocatable_gpus
        return remaining_gpus

    async def compute_action_mask(self):
        mask = []
        # refresh resources to get up-to-date free_memory_mb
        await self.update_resources_async()

        cur_cpu_1, cur_cpu_2 = self.current_deploy[:2]
        cur_gpu_1, cur_gpu_2, cur_gpu_3, cur_gpu_4 = self.current_deploy[2:6]

        # Capacity-aware mask: read remaining GPU units from ConfigMap (fallback to ratio * vGPU via node API)
        rem_units = None
        try:
            cm = self.v1.read_namespaced_config_map(name="remaining-gpu", namespace=self.namespace)
            data_dict = cm.data or {}
            val = data_dict.get(self.node_name)
            if val is not None:
                try:
                    rem_units = max(0, int(val))
                except Exception:
                    rem_units = None
        except Exception:
            rem_units = None
        if rem_units is None:
            try:
                frac = await asyncio.to_thread(self.get_remaining_gpus, self.node_name)
                if frac is None:
                    rem_units = 0
                else:
                    rem_units = max(0, int(float(self.vGPU) * float(frac)))
            except Exception:
                rem_units = 0

        busy_1 = await self.is_pod_busy(1)
        busy_2 = await self.is_pod_busy(2)
        busy_3 = await self.is_pod_busy(3)
        busy_4 = await self.is_pod_busy(4)

        # cooldown flags
        cd_1 = self.cooldown_info.get("llm1", {}).get("remaining_steps", 0) > 0
        cd_2 = self.cooldown_info.get("llm2", {}).get("remaining_steps", 0) > 0
        cd_3 = self.cooldown_info.get("llm3", {}).get("remaining_steps", 0) > 0
        cd_4 = self.cooldown_info.get("llm4", {}).get("remaining_steps", 0) > 0

        for combo in self.action_combo:
            # explicit KEEP action always feasible
            if any(v < 0 for v in combo):
                mask.append(1)
                continue
            # combo: [cpu_1, cpu_2, gpu_1, gpu_2, gpu_3, gpu_4]
            cpu_1, cpu_2, gpu_1, gpu_2, gpu_3, gpu_4 = combo
            feasible = True  
            # --- LLM1 ---
            if busy_1:
                if (cpu_1 != cur_cpu_1) or (gpu_1 != cur_gpu_1):
                    feasible = False
            if feasible and cd_1:
                if (cpu_1 != cur_cpu_1) or (gpu_1 != cur_gpu_1):
                    feasible = False
            # --- LLM2 ---
            if busy_2:
                if (cpu_2 != cur_cpu_2) or (gpu_2 != cur_gpu_2):
                    feasible = False
            if feasible and cd_2:
                if (cpu_2 != cur_cpu_2) or (gpu_2 != cur_gpu_2):
                    feasible = False
            # --- LLM3 ---
            if busy_3:
                if (gpu_3 != cur_gpu_3):
                    feasible = False
            if feasible and cd_3:
                if (gpu_3 != cur_gpu_3):
                    feasible = False
            # --- LLM4 ---
            if busy_4:
                if (gpu_4 != cur_gpu_4):
                    feasible = False
            if feasible and cd_4:
                if (gpu_4 != cur_gpu_4):
                    feasible = False

            # --- Capacity check on GPU additions ---
            if feasible and rem_units is not None:
                cur_gpu_bits = [cur_gpu_1, cur_gpu_2, cur_gpu_3, cur_gpu_4]
                new_gpu_bits = [gpu_1, gpu_2, gpu_3, gpu_4]
                d_add = 0
                for c, n in zip(cur_gpu_bits, new_gpu_bits):
                    if (n == 1) and (c == 0):
                        d_add += 1
                if d_add > rem_units:
                    feasible = False

            # --- Memory feasibility: ensure added memory fits into free MB ---
            if feasible and getattr(self, 'free_memory_mb', None) is not None:
                def gi_to_mb(s):
                    try:
                        return float(s.replace('Gi','')) * 1024.0
                    except Exception:
                        return 0.0
                req_mb = [gi_to_mb(x) for x in self.memory_require]
                cur_on = [int(cur_cpu_1 or cur_gpu_1), int(cur_cpu_2 or cur_gpu_2), int(cur_gpu_3), int(cur_gpu_4)]
                new_on = [int((cpu_1==1) or (gpu_1==1)), int((cpu_2==1) or (gpu_2==1)), int(gpu_3==1), int(gpu_4==1)]
                cur_mem = sum(req_mb[i] for i in range(4) if cur_on[i]==1)
                new_mem = sum(req_mb[i] for i in range(4) if new_on[i]==1)
                delta_mb = new_mem - cur_mem
                if delta_mb > self.free_memory_mb + 1e-6:
                    feasible = False

            mask.append(1 if feasible else 0)
        # Fallback: if no feasible action, allow a no-op/nearest configuration
        if sum(mask) == 0 and len(self.action_combo) > 0:
            try:
                cur = self.current_deploy
                exact_idx = None
                for i, combo in enumerate(self.action_combo):
                    if combo == cur:
                        exact_idx = i
                        break
                if exact_idx is not None:
                    mask[exact_idx] = 1
                else:
                    best_i, best_d = 0, 1e9
                    for i, combo in enumerate(self.action_combo):
                        try:
                            d = sum(1 for a,b in zip(combo, cur) if a != b)
                        except Exception:
                            d = 999
                        if d < best_d:
                            best_d, best_i = d, i
                    mask[best_i] = 1
            except Exception:
                pass
        return mask

    async def generate_state(self):
        self.pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={self.node_name}')
        
        pod_status = [0.0] * 4
        # for pod in self.pods.items:
        #     if pod.metadata.labels:
        #         for i in range(1, 5):
        #             pod_label = f"llm{i}-{self.node_name}"
        #             if pod.metadata.labels.get('llm') == pod_label:
        #                 pod_status[i - 1] = 1

        for pod in self.pods.items:
            if not pod.metadata.labels:
                continue

            llm_label = pod.metadata.labels.get("llm")
            for i in range(1, 5):
                if llm_label == f"llm{i}-{self.node_name}":
                    score = 0.0
                    if pod.metadata.deletion_timestamp:
                        score = 0.5
                    elif pod.status.phase == "Running":
                        all_ready = True
                        if pod.status.container_statuses:
                            all_ready = all(cs.ready for cs in pod.status.container_statuses)
                        score = 1.0 if all_ready else 0.5
                    elif pod.status.phase == "Pending":
                        score = 0.5
                    pod_status[i - 1] = score 
                    break

        pipeline = [
            {
                "$group": {
                    "_id": "$metadata.llm_anchor", 
                    "total_size": {
                        "$sum": {
                            "$toDouble": "$metadata.size" 
                        }
                    }
                }
            }
        ]
        prompt_sizes = {"1": 0, "2": 0, "3": 0, "4": 0}

        async for result in self.prompt_col.aggregate(pipeline):
            llm_anchor = result['_id']
            if llm_anchor not in ("1","2","3","4"):
                continue
            total_size = result['total_size']
            # Use raw aggregated size as workload so that controllers see
            # absolute queue pressure instead of a normalised value.
            prompt_sizes[llm_anchor] = float(total_size)
        
        logger.debug(f"prompt_sizes={prompt_sizes}")

        workload = list(prompt_sizes.values())

        await self.update_resources_async()

        logger.debug(f"pod_status={pod_status} (len={len(pod_status)})")
        logger.debug(f"workload={workload} (len={len(workload)})")
        logger.debug(f"current_resource={self.current_resource} (len={len(self.current_resource)})")

        self.state = pod_status + workload + self.current_resource
        # e.g. next_state=[1.0, 0.5, 1.0, 0.0, 0.18, 0.32, 0, 0, 0.97835071835195, 0.00081324157860549, 0.5, 1.0]
        return self.state

    async def step(self, action):
        for llm_key in self.cooldown_info:
            if self.cooldown_info[llm_key]["remaining_steps"] > 0:
                self.cooldown_info[llm_key]["remaining_steps"] -= 1

        state = await self.generate_state()
        await self.execute_action(action)
        next_state = await self.generate_state()
        return state, next_state

    async def execute_action(self, action):
        self.deploy_plan = self.action_combo[action]
        # explicit KEEP: no changes
        if any(v < 0 for v in self.deploy_plan):
            logger.debug("[KEEP] Keeping current deployment; no changes applied.")
            return
        logger.debug(f"execute_action called with deploy_plan={self.deploy_plan}")

        cpu_bits_new = self.deploy_plan[:2]
        gpu_bits_new = self.deploy_plan[2:]

        cpu_bits_cur = self.current_deploy[:2]
        gpu_bits_cur = self.current_deploy[2:]

        for llm_anchor in [1, 2, 3, 4]:
            llm_key = f"llm{llm_anchor}"
            pod_name = f"llm{llm_anchor}-{self.node_name}"

            if self.cooldown_info[llm_key]["remaining_steps"] > 0:
                logger.info(f"[COOLDOWN] {llm_key} has {self.cooldown_info[llm_key]['remaining_steps']} steps left, skip.")
                continue

            if await self.is_pod_pending(llm_anchor):
                logger.info(f"[PENDING] {pod_name} is in Pending, skip.")
                continue

            if llm_anchor == 1:
                want_cpu = (cpu_bits_new[0] == 1)
                cur_cpu  = (cpu_bits_cur[0] == 1)
                want_gpu = (gpu_bits_new[0] == 1)
                cur_gpu  = (gpu_bits_cur[0] == 1)

            elif llm_anchor == 2:
                want_cpu = (cpu_bits_new[1] == 1)
                cur_cpu  = (cpu_bits_cur[1] == 1)
                want_gpu = (gpu_bits_new[1] == 1)
                cur_gpu  = (gpu_bits_cur[1] == 1)

            elif llm_anchor == 3:
                want_cpu = False   
                cur_cpu  = False
                want_gpu = (gpu_bits_new[2] == 1)
                cur_gpu  = (gpu_bits_cur[2] == 1)

            else: 
                want_cpu = False
                cur_cpu  = False
                want_gpu = (gpu_bits_new[3] == 1)
                cur_gpu  = (gpu_bits_cur[3] == 1)

            if want_cpu:
                if cur_cpu:
                    logger.debug(f"[SKIP] {pod_name} already on CPU.")
                else:
                    logger.info(f"[DEPLOY] {pod_name} -> CPU")
                    try:
                        await self.update_pod(
                            namespace=self.namespace,
                            cpu=f"{int(self.cpu_upperlimit)-5}",
                            gpu="0",
                            memory=self.memory_require[llm_anchor-1],
                            llm_anchor=llm_anchor,
                            node_selector={"workerid": f"{self.node_name}"}
                        )

                        if llm_anchor == 1:
                            self.current_deploy[0] = 1  
                            self.current_deploy[2] = 0 
                        elif llm_anchor == 2:
                            self.current_deploy[1] = 1  
                            self.current_deploy[3] = 0 
                    except Exception as e:
                        logger.error(f"[DEPLOY-ERROR] CPU deployment failed for {pod_name}: {e}")

            elif want_gpu:
                if cur_gpu:
                    logger.debug(f"[SKIP] {pod_name} already on GPU.")
                else:
                    logger.info(f"[DEPLOY] {pod_name} -> GPU")
                    try:
                        await self.update_pod(
                            namespace=self.namespace,
                            cpu="1",
                            gpu="1",
                            memory=self.memory_require[llm_anchor-1],
                            llm_anchor=llm_anchor,
                            node_selector={"workerid": f"{self.node_name}"}
                        )

                        if llm_anchor == 1:
                            self.current_deploy[0] = 0  
                            self.current_deploy[2] = 1  
                        elif llm_anchor == 2:
                            self.current_deploy[1] = 0  
                            self.current_deploy[3] = 1  
                        elif llm_anchor == 3:
                            self.current_deploy[4] = 1 
                        else:
                            self.current_deploy[5] = 1 
                    except Exception as e:
                        logger.error(f"[DEPLOY-ERROR] GPU deployment failed for {pod_name}: {e}")

            else: # doesnt' need anything
                if cur_cpu or cur_gpu: # but still using resources
                    logger.info(f"[DELETE] {pod_name} (was CPU={cur_cpu}, GPU={cur_gpu})")
                    try:
                        await asyncio.to_thread(
                            self.api_instance.delete_namespaced_deployment,
                            name=pod_name,
                            namespace=self.namespace,
                            body=V1DeleteOptions(grace_period_seconds=0, propagation_policy='Background')
                        )
                        await self.wait_for_deployment_deleted(self.namespace, pod_name)
                        self.cooldown_info[llm_key]["remaining_steps"] = self.cooldown_period
                        self.cooldown_info[llm_key]["last_action"] = "delete"

                        if llm_anchor == 1:
                            self.current_deploy[0] = 0
                            self.current_deploy[2] = 0
                        elif llm_anchor == 2:
                            self.current_deploy[1] = 0
                            self.current_deploy[3] = 0
                        elif llm_anchor == 3:
                            self.current_deploy[4] = 0
                        else:
                            self.current_deploy[5] = 0
                    except self.client.exceptions.ApiException as e:
                        if e.status == 404:
                            logger.info(f"[DELETE-SKIP] {pod_name} not found.")
                        else:
                            logger.error(f"[DELETE-ERROR] {pod_name}: {e}")
                else:
                    logger.debug(f"[SKIP] {pod_name} already not deployed.")

        logger.debug(f"[INFO] Updated self.current_deploy = {self.current_deploy}")

        used_gpu = sum(self.current_deploy[2:6])
        await self.update_gpu_usage_in_configmap(used_gpu)
    
    async def update_gpu_usage_in_configmap(self, used_gpu: int):
        configmap_name = "remaining-gpu"
        namespace = self.namespace
        node_key = self.node_name 

        patch_body = {
            "data": {
                node_key: str(self.vGPU-used_gpu)
            }
        }

        try:
            await asyncio.to_thread(
                self.v1.patch_namespaced_config_map,
                configmap_name,
                namespace,
                patch_body
            )
            logger.debug(f"[CONFIGMAP] Patched {configmap_name} in {namespace}: set data['{node_key}']={self.vGPU-used_gpu}")
        except self.client.exceptions.ApiException as e:
            if e.status == 404:
                cm_body = self.client.V1ConfigMap(
                    metadata= self.client.V1ObjectMeta(name=configmap_name),
                    data={
                        node_key: str(self.vGPU-used_gpu)
                    }
                )
                await asyncio.to_thread(
                    self.v1.create_namespaced_config_map,
                    namespace,
                    cm_body
                )
                logger.info(f"[CONFIGMAP] Created new ConfigMap {configmap_name} in {namespace} with data['{node_key}']={self.vGPU-used_gpu}")
            else:
                logger.error(f"[CONFIGMAP] Failed to patch or create {configmap_name} in {namespace}: {e}")

    async def wait_for_deployment_deleted(self, namespace, name, check_interval=5, timeout=120):
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                await asyncio.to_thread(self.api_instance.read_namespaced_deployment, name=name, namespace=namespace)

                deployment = await asyncio.to_thread(
                    self.api_instance.read_namespaced_deployment, 
                    name=name, 
                    namespace=namespace
                )

                logger.debug("Deployment %s is still present. Checking more details...", name)

                if deployment.metadata.finalizers:
                    logger.debug("Deployment finalizers: %s", deployment.metadata.finalizers)

                if deployment.status.conditions:
                    for cond in deployment.status.conditions:
                        logger.debug("Deployment condition type=%s status=%s reason=%s message=%s",
                                    cond.type, cond.status, cond.reason, cond.message)

                # print the status of the corrsponding status of replicaset / pods
                await self.debug_deployment_children(namespace, name)

            except self.client.exceptions.ApiException as e:
                if e.status == 404:
                    return
                else:
                    print(f"[WARN] Unexpected error when checking deployment deletion: {e}")
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"Deployment {name} in {namespace} not deleted after {timeout} seconds.")

    async def debug_deployment_children(self, namespace, deployment_name):
        """Inspect ReplicaSets owned by a deployment."""
        rs_list = await asyncio.to_thread(self.api_instance.list_namespaced_replica_set, namespace=namespace)
        for rs in rs_list.items:
            owners = rs.metadata.owner_references or []
            for owner in owners:
                if owner.kind == "Deployment" and owner.name == deployment_name:
                    logger.debug("ReplicaSet %s owned by Deployment %s, readyReplicas=%s, replicas=%s",
                                rs.metadata.name, deployment_name, rs.status.ready_replicas, rs.status.replicas)
                    await self.debug_replicaset_pods(namespace, rs.metadata.name)

    async def debug_replicaset_pods(self, namespace, rs_name):
        """Inspect pods owned by a ReplicaSet."""
        pod_list = await asyncio.to_thread(self.v1.list_namespaced_pod, namespace=namespace, label_selector=f"ownerReferences.name={rs_name}")
        for pod in pod_list.items:
            logger.debug("Pod %s under RS %s, phase=%s, finalizers=%s",
                        pod.metadata.name, rs_name, pod.status.phase, pod.metadata.finalizers)

    async def create_deployment(self, namespace, cpu, gpu, memory, llm_anchor, node_selector=None):
        def sync_create_deployment():
            env_vars = [
                self.client.V1EnvVar(name="USE_GPU", value=gpu),
                # self.client.V1EnvVar(name="NODE_NAME", value=self.node_name),
                self.client.V1EnvVar(name="NODE_IP", value=self.node_ip)
                ]
            
            volume_mounts = [
                self.client.V1VolumeMount(
                    name='local-prompts',
                    mount_path='/app/prompt_queue'
                ),
                self.client.V1VolumeMount(
                    name='local-outputs',
                    mount_path='/app/output_local'
                ),
                self.client.V1VolumeMount(
                    name='local-output-relay',
                    mount_path='/app/output_relay'
                ),
                self.client.V1VolumeMount(
                    name='local-current-prompt',
                    mount_path='/app/current_prompt'
                )
            ]

            volumes = [
                self.client.V1Volume(
                    name='local-prompts',
                    host_path=self.client.V1HostPathVolumeSource(
                        path='/var/lib/ma-edge-llm/prompt_queue',
                        type='DirectoryOrCreate'  # Creates the directory if it doesn't exist
                    )
                ),
                self.client.V1Volume(
                    name='local-outputs',
                    host_path=self.client.V1HostPathVolumeSource(
                        path='/var/lib/ma-edge-llm/output_local',
                        type='DirectoryOrCreate'
                    )
                ),
                self.client.V1Volume(
                    name='local-output-relay',
                    host_path=self.client.V1HostPathVolumeSource(
                        path='/var/lib/ma-edge-llm/output_relay',
                        type='DirectoryOrCreate'
                    )
                ),
                self.client.V1Volume(
                    name='local-current-prompt',
                    host_path=self.client.V1HostPathVolumeSource(
                        path='/var/lib/ma-edge-llm/current_prompt',
                        type='DirectoryOrCreate'
                    )
                )
            ]

            # Define resource requests and limits
            resource_limits = {"cpu": cpu, "memory": memory}
            resource_requests = {"cpu": cpu, "memory": memory}
            
            if gpu and gpu != "0":  
                resource_limits["nvidia.com/gpu"] = gpu
                resource_requests["nvidia.com/gpu"] = gpu

            # Define Pod template
            image_prefix = os.getenv("IMAGE_PREFIX", "local/elm")
            container = self.client.V1Container(
                name=f'llm{llm_anchor}-{self.node_name}',
                image=f'{image_prefix}-lm{llm_anchor}',
                env=env_vars,
                volume_mounts=volume_mounts,
                image_pull_policy="IfNotPresent",
                command=["/bin/bash", "-c", f"python /app/inference_lm{llm_anchor}.py"],
                resources=self.client.V1ResourceRequirements(
                    limits=resource_limits,
                    requests=resource_requests,
                )
            )

            pod_spec = self.client.V1PodSpec(
                containers=[container],
                image_pull_secrets=[self.client.V1LocalObjectReference(name="llm-docker")],
                # restart_policy="Never", this is not appliable for Deployment
                volumes=volumes,
                service_account_name="custom-sa"
            )

            llm_labels = {
                "llm": f"llm{llm_anchor}-{self.node_name}",
                "llm_mode": "gpu" if gpu and gpu != "0" else "cpu",
            }
            pod_template = self.client.V1PodTemplateSpec(
                metadata=self.client.V1ObjectMeta(labels=llm_labels),
                spec=pod_spec
            )

            if node_selector:
                pod_template.spec.node_selector = node_selector

            deployment_spec = self.client.V1DeploymentSpec(
                replicas=1,  # Start with 1 replica
                template=pod_template,
                selector={'matchLabels': {'llm': f"llm{llm_anchor}-{self.node_name}"}}
            )

            deployment = self.client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=self.client.V1ObjectMeta(name=f'llm{llm_anchor}-{self.node_name}'),
                spec=deployment_spec
            )

            try:
                
                self.api_instance.create_namespaced_deployment(namespace=namespace, body=deployment)
                # print(f"Deployment llm{llm_anchor} created.")
            except self.client.exceptions.ApiException as e:
                print(f"Exception when creating deployment: {e}")

        await asyncio.to_thread(sync_create_deployment)

    async def update_pod(self, namespace, cpu, gpu, memory, llm_anchor, node_selector=None):
        try:
            deployment = await asyncio.to_thread(
                self.api_instance.read_namespaced_deployment,
                name=f'llm{llm_anchor}-{self.node_name}',
                namespace=namespace
            )
        except self.client.exceptions.ApiException as e:
            if e.status == 404:
                await self.create_deployment(namespace=namespace, cpu=cpu, gpu=gpu, memory=memory, llm_anchor=llm_anchor, node_selector=node_selector)
                return
            else:
                raise
        
        llm_key = f"llm{llm_anchor}"
        current_gpu = False
        for container in deployment.spec.template.spec.containers:
            resources = container.resources
            if resources.limits and "nvidia.com/gpu" in resources.limits:
                current_gpu = True
                break

        if gpu != "0":
            if not current_gpu:
                # await asyncio.to_thread(
                #     self.api_instance.delete_namespaced_deployment,
                #     name=f'llm{llm_anchor}',
                #     namespace=namespace
                # )
                await asyncio.to_thread(
                    self.api_instance.delete_namespaced_deployment,
                    name=f'llm{llm_anchor}-{self.node_name}',
                    namespace=namespace,
                    body=V1DeleteOptions(grace_period_seconds=0, propagation_policy='Background')
                )
                await self.wait_for_deployment_deleted(namespace, f'llm{llm_anchor}-{self.node_name}')
                await self.create_deployment(namespace=namespace, cpu=cpu, gpu=gpu, memory=memory, llm_anchor=llm_anchor, node_selector=node_selector)
                self.cooldown_info[llm_key]["remaining_steps"] = self.cooldown_period
                self.cooldown_info[llm_key]["last_action"] = "cpu_to_gpu"
            else:
                pass
        else:
            if current_gpu:
                # await asyncio.to_thread(
                #     self.api_instance.delete_namespaced_deployment,
                #     name=f'llm{llm_anchor}',
                #     namespace=namespace
                # )
                await asyncio.to_thread(
                    self.api_instance.delete_namespaced_deployment,
                    name=f'llm{llm_anchor}-{self.node_name}',
                    namespace=namespace,
                    body=V1DeleteOptions(grace_period_seconds=0, propagation_policy='Background')
                )
                await self.wait_for_deployment_deleted(namespace, f'llm{llm_anchor}-{self.node_name}')
                await self.create_deployment(namespace=namespace, cpu=cpu, gpu='0', memory=memory, llm_anchor=llm_anchor, node_selector=node_selector)
                self.cooldown_info[llm_key]["remaining_steps"] = self.cooldown_period
                self.cooldown_info[llm_key]["last_action"] = "gpu_to_cpu"
            else:
                patch = {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": f'llm{llm_anchor}-{self.node_name}',
                                    "resources": {
                                        "limits": {
                                            "cpu": cpu,
                                            "memory": memory
                                        },
                                        "requests": {
                                            "cpu": cpu,
                                            "memory": memory
                                        }
                                    }
                                }]
                            }
                        }
                    }
                }
                await asyncio.to_thread(
                    self.api_instance.patch_namespaced_deployment,
                    name=f'llm{llm_anchor}-{self.node_name}',
                    namespace=namespace,
                    body=patch
                )
            
    def _load_anchor_stats(self, csv_path):
        import csv
        if self._anchor_stats is not None:
            return self._anchor_stats 

        dist_map = {"1": [], "2": [], "3": [], "4": []}
        with open(csv_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            for row in reader:
                if not row:
                    continue
                key = row[0].strip()
                ip_part, llm_part = key.split("-llm", 1)
                if ip_part.strip() == self.node_ip and llm_part.strip() in dist_map:
                    # Collect latency samples.
                    dist_map[llm_part.strip()].extend(
                        float(v) for v in row[1:] if v and v.lower() != "none"
                    )

        anchor_p10, anchor_p90 = {}, {}
        for a_id, arr in dist_map.items():
            if len(arr) < 2:
                anchor_p10[a_id], anchor_p90[a_id] = 10.0, 900.0
            else:
                arr_np = np.array(arr)
                p10, p90 = np.percentile(arr_np, 10), np.percentile(arr_np, 90)
                if p10 < 1e-5 or (p90 - p10) < 1e-5:
                    anchor_p10[a_id], anchor_p90[a_id] = 10.0, 900.0
                else:
                    anchor_p10[a_id], anchor_p90[a_id] = p10, p90

        self._anchor_stats = (anchor_p10, anchor_p90)
        return self._anchor_stats

    async def reward_retrieval(self, csv_path="/app/sadrl/latency_data.csv"):
        anchor_p10, anchor_p90 = self._load_anchor_stats(csv_path)

        def log_reward(T, tmin, tmax):
            if T <= tmin:
                return 1.0
            if T >= tmax:
                return 0.0
            val = 1.0 - (math.log(T) - math.log(tmin)) / (math.log(tmax) - math.log(tmin))
            return max(0.0, min(1.0, val))

        query = {"slot_anchor": {"$exists": True}}

        projection = {
            "slot_anchor": 1,
            "1.skip": 1, "1.fail_reason": 1, "1.single_llm_delay": 1,
            "2.skip": 1, "2.fail_reason": 1, "2.single_llm_delay": 1,
            "3.skip": 1, "3.fail_reason": 1, "3.single_llm_delay": 1,
            "4.skip": 1, "4.fail_reason": 1, "4.single_llm_delay": 1,
            "_id": 0
        }
        cursor = self.latency_col.find(query, projection)

        records = []
        async for doc in cursor:
            slot_anchor = int(doc["slot_anchor"])
            all_final = True
            valid_delays = []
            for anchor_id in ["1", "2", "3", "4"]:
                anchor_dict = doc.get(anchor_id, {})
                skip_flag = anchor_dict.get("skip", False)
                delay_val = anchor_dict.get("single_llm_delay", None)

                if (not skip_flag) and (delay_val is None):
                    all_final = False
                    break

            if not all_final:
                continue

            for anchor_id in ["1", "2", "3", "4"]:
                anchor_dict = doc.get(anchor_id, {})
                skip_flag = anchor_dict.get("skip", False)
                delay_val = anchor_dict.get("single_llm_delay", None)
                if (not skip_flag) and (delay_val is not None):
                    valid_delays.append(log_reward(delay_val, anchor_p10[anchor_id], anchor_p90[anchor_id]))

            if len(valid_delays) == 0:
                outside_reward = 0.0
            else:
                outside_reward = sum(valid_delays) / len(valid_delays)

            records.append({
                "slot_anchor": slot_anchor,
                "doc": doc, 
                "outside_reward": outside_reward
            })

        if not records:
            return []

        grouped_results = {}
        for item in records:
            slot_anchor = item["slot_anchor"]
            group_key = slot_anchor // self.length
            grouped_results.setdefault(group_key, []).append(item)

        reward_col = self.mongo["reward"][f"{self.node_ip.replace('.', '_')}"]
        retrievaled_reward = []

        for group_key, group_data in grouped_results.items():
            group_key_str = str(group_key)

            doc_found = await reward_col.find_one(
                {"local_slot_anchor": group_key_str},
                {"processed": 1}
            )
            if doc_found and doc_found.get("processed") is True:
                continue

            for doc_item in group_data:
                slot_anchor = doc_item["slot_anchor"]
                subfield_key = str(slot_anchor)
                outside_reward = doc_item["outside_reward"]
                original_doc = doc_item["doc"]

                anchors_info = {}
                for anchor_id in ["1","2","3","4"]:
                    anchor_dict = original_doc.get(anchor_id, {})
                    skip_flag = anchor_dict.get("skip", False)
                    fail_reason = anchor_dict.get("fail_reason", None)
                    delay_val = anchor_dict.get("single_llm_delay", None)

                    if skip_flag:
                        anchors_info[f"llm{anchor_id}"] = "skip"
                    elif fail_reason == "timeout":
                        anchors_info[f"llm{anchor_id}"] = "time_out"
                    elif delay_val is not None:
                        anchors_info[f"llm{anchor_id}"] = delay_val
                        anchors_info[f"score{anchor_id}"] = log_reward(delay_val, anchor_p10[anchor_id], anchor_p90[anchor_id])
                    else:
                        anchors_info[f"llm{anchor_id}"] = "unknown"

                anchors_info["outside_reward"] = outside_reward

                await reward_col.update_one(
                    {"local_slot_anchor": group_key_str},
                    {
                        "$set": {
                            subfield_key: anchors_info,
                            "local_slot_anchor": group_key_str
                        }
                    },
                    upsert=True
                )

                # Mirror per-slot node reward into the aggregate reward stream.
                await self._safe_upsert_ma_reward(slot_anchor, outside_reward)

            if len(group_data) == self.length:
                valid_slots = [x for x in group_data if x["outside_reward"] > 0]
                if not valid_slots:
                    group_avg_reward = 0.0
                else:
                    sum_reward = sum(x["outside_reward"] for x in valid_slots)
                    group_avg_reward = sum_reward / len(valid_slots)

                await reward_col.update_one(
                    {"local_slot_anchor": group_key_str},
                    {
                        "$set": {
                            "reward": group_avg_reward,
                            "processed": True
                        }
                    }
                )
                retrievaled_reward.append([group_key, group_avg_reward])

        return retrievaled_reward

    def store_data(self, data_list, new_data, number): # changed to only save number amount, no more
        if len(data_list) < number:
            data_list.append(new_data)
        else:
            pass
        # if len(data_list) > number:
        #     del data_list[0]

    def bottom_avg(self, data_list):
        new_data_list = copy.deepcopy(data_list)
        new_data_list.sort()
        n = len(new_data_list)
        percent = int(n * 0.1)
        avg = 0
        minimum = new_data_list[0]
        if percent == 0:
            avg = minimum
        else:
            avg = sum(new_data_list[0:percent]) / percent
        return avg

    async def _safe_upsert_ma_reward(self, slot_anchor: int, outside_reward: float) -> None:
        """Upsert per-slot reward into the aggregate reward collection.
        Enforces one doc per slot via _id=str(slot_anchor).
        """
        filter_doc = {"_id": str(slot_anchor)}
        update_doc = {
            "$set": {
                f"{self.node_ip.replace('.', '_')}.reward": float(outside_reward),
                "slot_anchor": str(slot_anchor)
            }
        }
        try:
            await self.reward_ma_col.update_one(filter_doc, update_doc, upsert=True)
        except DuplicateKeyError:
            await self.reward_ma_col.update_one(filter_doc, update_doc, upsert=False)
        except Exception as _e:
            logger.debug(f"ma_reward upsert failed for slot {slot_anchor}: {_e}")
    
    def reset(self):
        # self.local_slot_anchor = 0
        deployments = self.api_instance.list_deployment_for_all_namespaces()
        pattern = re.compile(r"^llm\d+$")
        for deployment in deployments.items:
            name = deployment.metadata.name
            namespace = deployment.metadata.namespace
            if pattern.match(name):
                try:
                    self.api_instance.delete_namespaced_deployment(name, namespace)
                except self.client.exceptions.ApiException as e:
                    if e.status == 404:
                        pass
                    else:
                        print(f"Failed to delete deployment {name} in {namespace}: {e}")
        # return np.array(self.state, dtype=np.float32)
