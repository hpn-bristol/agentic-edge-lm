import subprocess
import numpy as np
from kubernetes import client, config
from kubernetes.stream import stream
import json
import time
import random
import datetime
import multiprocessing
import os
import math
from pymongo import MongoClient
import copy
import requests
import re
import asyncio
import httpx
from motor.motor_asyncio import AsyncIOMotorClient

# Optional 3-phase dynamic workload schedule. Activated only when
# WORKLOAD_PROFILE=stress3phase; otherwise the generator uses request_range.
# Each entry: (slot_lo, slot_hi, [[min,max] x4]) for LM1..LM4.
_STRESS3PHASE = [
    (0,    600,  [[0, 8],  [0, 8],  [0, 8],  [0, 8]]),   # balanced
    (600,  800,  [[0, 8],  [0, 8],  [6, 15], [0, 8]]),   # LM3-only surge
    (800,  1000, [[6, 15], [0, 6],  [0, 6],  [6, 15]]),  # LM1+LM4 surge
]


class EllmEnv():
    def __init__(self, dbname, request_range, length_of_slot, status):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.v1 = client.CoreV1Api() 
        self.appsv1 = client.AppsV1Api()

        service = self.v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        self.mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")

        self.nodes = []
        self.node_ips = []
        self.node_resources = {}

        sorted_nodes = sorted(self.v1.list_node().items, key=lambda node: (node.metadata.name != 'controller', self.natural_sort_key(node)))

        # Build 1:1 (node_name, ip) pairs; skip nodes without a single InternalIP
        pairs = []
        for node in sorted_nodes:
            labels = node.metadata.labels or {}
            if 'node-role.kubernetes.io/controlplane' in labels:
                continue
            hostname = node.metadata.name
            # collect InternalIP(s)
            ips = [addr.address for addr in (node.status.addresses or []) if addr.type == "InternalIP" and addr.address]
            if len(ips) != 1:
                # skip nodes with missing or multiple InternalIP to preserve shape consistency
                continue
            pairs.append((hostname, ips[0], node))

        for hostname, ip_addr, node in pairs:
            labels = node.metadata.labels or {}
            node_allocatable = node.status.allocatable

            self.nodes.append(hostname)
            self.node_ips.append(ip_addr)

            # Only rely on label nvidia.com/gpu.replicas as requested
            replicas_label = labels.get("nvidia.com/gpu.replicas") if labels else None
            try:
                gpu_replicas = int(replicas_label) if replicas_label and str(replicas_label).isdigit() else 0
            except Exception:
                gpu_replicas = 0

            self.node_resources[hostname] = {
                'max_cpu': int(node_allocatable.get("cpu")),
                'max_memory': int(node_allocatable.get("memory")[:-2]),
                'max_storage': int(node_allocatable.get("ephemeral-storage")[:-2]),
                'max_gpu': gpu_replicas,
                'max_pods': int(node_allocatable.get("pods"))
            }
            # print(f"[GPU-REPLICAS] node={hostname} replicas={gpu_replicas}")
        self.request_list = None
        self.resource_col = self.mongo["k8smonitor"][dbname]
        self.slot_anchor = 0
        self.incre_from = 0
        # Reward DB handles:
        # - Aggregated per-slot reward in reward.ma_reward
        # - Per-node local rewards in reward.{ip_underscore}
        # - Consumed marker collection to avoid reusing the same slot multiple times
        self.reward_db = self.mongo["reward"]
        self.reward_col = self.reward_db["ma_reward"]
        self.consumed_col = self.reward_db["consumed_slots"]
        
        self.status = status
        self.dbname = dbname
        self.request_range = request_range
        self.length_of_slot = length_of_slot
        self.llm_amount = len(self.request_range)
        self.vGPU = 4
        self.slot_llm_map = {}
        # Cache last known resources to survive temporary monitoring outages
        self.last_resource = None  # shape: [[cpu_free, mem_norm, gpu_frac] per node]
        # Monitoring health snapshot (updated by update_resources)
        self.monitor_status = {
            'doc_ok': False,
            'using_cached_doc': False,
            'nodes_missing': [],
            'gpu_configmap_missing': [],
            'used_cached_cpu_mem_nodes': [],
            'used_cached_gpu_nodes': []
        }
        # One-shot transition logging state
        self._prev_degraded = None
        self._degraded_since_slot = None
        self._degraded_since_time = None
        self.verbose_monitor = True
        # One-time diagnostic flag for global_state segment lengths.
        self._gs_debug_printed = False

    def _log_monitor_transition(self, status):
        degraded = (not status.get('doc_ok')) or len(status.get('nodes_missing', [])) > 0
        now = time.time()
        # First observation: initialize state and optionally print degraded
        if self._prev_degraded is None:
            self._prev_degraded = degraded
            if degraded and self.verbose_monitor:
                self._degraded_since_time = now
                self._degraded_since_slot = self.slot_anchor
                print(
                    f"[monitor] degraded: doc_ok={status.get('doc_ok')} using_cache={status.get('using_cached_doc')} "
                    f"missing_nodes={len(status.get('nodes_missing', []))} "
                    f"gpu_cm_missing={len(status.get('gpu_configmap_missing', []))} "
                    f"slot={self.slot_anchor}"
                )
            return
        # Transition detection
        if self._prev_degraded != degraded:
            if degraded:
                # Entering degraded
                self._degraded_since_time = now
                self._degraded_since_slot = self.slot_anchor
                if self.verbose_monitor:
                    print(
                        f"[monitor] degraded: doc_ok={status.get('doc_ok')} using_cache={status.get('using_cached_doc')} "
                        f"missing_nodes={len(status.get('nodes_missing', []))} "
                        f"gpu_cm_missing={len(status.get('gpu_configmap_missing', []))} "
                        f"slot={self.slot_anchor}"
                    )
            else:
                # Recovering
                dur_sec = int(now - (self._degraded_since_time or now))
                slots = max(0, self.slot_anchor - (self._degraded_since_slot or self.slot_anchor))
                if self.verbose_monitor:
                    print(
                        f"[monitor] recovered: missing_nodes=0 slot={self.slot_anchor} "
                        f"duration={dur_sec}s slots={slots}"
                    )
            self._prev_degraded = degraded

    @staticmethod
    async def post_with_retries_async(url, json, max_retries=20, retry_delay=3, timeout=900):
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(url, json=json, timeout=timeout)
                    if response.status_code == 200:
                        return response
                    else:
                        print(f"[{url}] Non-200 status: {response.status_code}, attempt {attempt+1}")
                except httpx.RequestError as e:
                    print(f"[{url}] Attempt {attempt+1} failed with error: {e}")
                await asyncio.sleep(retry_delay)
        return None

    def natural_sort_key(self, node):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', node.metadata.name)]

    # def get_remaining_gpus(self, node_name):
    #     node = self.v1.read_node(node_name)
    #     allocatable_gpus = node.status.allocatable.get('nvidia.com/gpu')
    #     if allocatable_gpus:
    #         allocatable_gpus = int(allocatable_gpus)
    #     else:
    #         # print(f"Node {node_name} does not have GPUs or the nvidia.com/gpu resource is not found.")
    #         return 0
    #     requested_gpus = 0
    #     pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node_name}')
    #     for pod in pods.items:
    #         for container in pod.spec.containers:
    #             if container.resources and container.resources.requests:
    #                 gpu_request = container.resources.requests.get('nvidia.com/gpu')
    #                 if gpu_request:
    #                     requested_gpus += int(gpu_request)
    #     remaining_gpus = (allocatable_gpus - requested_gpus) / allocatable_gpus
    #     return remaining_gpus

    def get_remaining_gpus(self, node_name, namespace="allworkers"):
        configmap_name = "remaining-gpu"

        try:
            cm = self.v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
            data_dict = cm.data or {}

            usage_str = data_dict.get(node_name)
            if usage_str is not None:
                return int(usage_str)
            else:
                return None

        except client.exceptions.ApiException as e:
            if e.status == 404:
                print(f"[WARN] ConfigMap {configmap_name} not found in namespace={namespace}.")
                return None
            else:
                print(f"[ERROR] Failed to read ConfigMap {configmap_name} in namespace={namespace}: {e}")
                return None

    async def update_resources(self):
        data = self.resource_col.find().sort("timestamp", -1).limit(1)
        docs = await data.to_list(length=1)

        new_resources = []
        # Reset status for this cycle
        status = {
            'doc_ok': False,
            'using_cached_doc': False,
            'nodes_missing': [],
            'gpu_configmap_missing': [],
            'used_cached_cpu_mem_nodes': [],
            'used_cached_gpu_nodes': []
        }
        # If no document at all: reuse last known snapshot if available, else optimistic defaults
        if not docs:
            if self.last_resource and len(self.last_resource) == len(self.nodes):
                self.current_resource = copy.deepcopy(self.last_resource)
                status['doc_ok'] = False
                status['using_cached_doc'] = True
                self.monitor_status = status
                # One-shot transition log
                self._log_monitor_transition(status)
                return
            else:
                self.current_resource = [[1.0, 1.0, 1.0] for _ in self.nodes]
                self.last_resource = copy.deepcopy(self.current_resource)
                status['doc_ok'] = False
                status['using_cached_doc'] = True
                self.monitor_status = status
                self._log_monitor_transition(status)
                return
        
        doc = docs[0]
        metrics = doc.get('metrics', {})
        # Pre-compute denominators to avoid recomputing inside loop
        max_cpu_cap = max(self.node_resources[n]['max_cpu'] for n in self.node_resources)
        max_mem_cap = max(self.node_resources[n]['max_memory'] for n in self.node_resources)

        for idx, node in enumerate(self.nodes):
            node_caps = self.node_resources.get(node, {})
            max_cpu = node_caps.get('max_cpu', 1) or 1

            m = metrics.get(node)
            if m is None:
                # Prefer cached values per-node to avoid spikes; otherwise neutral defaults
                if self.last_resource and idx < len(self.last_resource):
                    cpu_free, mem_norm, _gpu_last = self.last_resource[idx]
                else:
                    cpu_free, mem_norm = 1.0, 1.0
                status['nodes_missing'].append(node)
                status['used_cached_cpu_mem_nodes'].append(node)
            else:
                # CPU free scaled by per-cluster max capacity to keep values ~[0,1]
                cpu_util = float(m.get('cpu', 0.0))  # utilization %
                cpu_free = ((1.0 - (cpu_util / 100.0)) * max_cpu) / float(max(max_cpu_cap, 1))

                # Memory free normalized by cluster max capacity (keeps shape consistent with existing code)
                free_mem_mb = float(m.get('free_memory', 0.0)) / 1024.0  # bytes -> MB
                mem_norm = free_mem_mb / float(max(max_mem_cap, 1))

            # GPU remaining from ConfigMap; if None, fallback to cached gpu for this node; else 0
            remaining_units = self.get_remaining_gpus(node)
            if remaining_units is None:
                if self.last_resource and idx < len(self.last_resource):
                    gpu_frac = float(self.last_resource[idx][2])
                    status['used_cached_gpu_nodes'].append(node)
                else:
                    gpu_frac = 0.0
                status['gpu_configmap_missing'].append(node)
            else:
                try:
                    gpu_frac = float(remaining_units) / float(self.vGPU)
                except Exception:
                    gpu_frac = 0.0

            # Clamp to [0,1]
            cpu_free = max(0.0, min(1.0, cpu_free))
            mem_norm = max(0.0, min(1.0, mem_norm))
            gpu_frac = max(0.0, min(1.0, gpu_frac))

            new_resources.append([cpu_free, mem_norm, gpu_frac])

        self.current_resource = new_resources
        self.last_resource = copy.deepcopy(self.current_resource)
        status['doc_ok'] = True
        self.monitor_status = status
        # One-shot transition log
        self._log_monitor_transition(status)

    def get_monitor_status(self):
        return copy.deepcopy(self.monitor_status)

    def is_monitor_recovered(self):
        # Recovered when we have a doc and no nodes are missing
        ms = self.monitor_status
        return bool(ms.get('doc_ok')) and len(ms.get('nodes_missing', [])) == 0

    async def generate_global_state(self):
        # Ensure we have a request_list even in pure routing mode where
        # reset() does not eagerly generate requests. Fall back to the
        # current slot_llm_map so that slot 0 has a well-defined workload
        # vector before the first generate_requests_async() call.
        if self.request_list is None:
            self.request_list = []
            for ip in self.node_ips:
                self.request_list.append(self.slot_llm_map.get(ip, [0, 0, 0, 0]))

        normalized_request_list = [[value / r[1] for value, r in zip(request, self.request_range)] for request in self.request_list]
        requests = [item for sublist in normalized_request_list for item in sublist]

        await self.update_resources()
        resources = [item for sublist in self.current_resource for item in sublist]

        workloads = []
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
        for ip in self.node_ips:
            # initialize per-node prompt sizes with fixed string keys to keep a stable 4-dim order
            prompt_sizes = {"1": 0, "2": 0, "3": 0, "4": 0}
            prompt_col = self.mongo["prompt_recorder"][f"{ip.replace('.', '_')}"]
            cursor = prompt_col.aggregate(pipeline)

            async for result in cursor:
                # ensure key is a string and update only known anchors to avoid accidental extra dims
                try:
                    llm_anchor = str(result.get('_id'))
                except Exception:
                    llm_anchor = None
                if llm_anchor in prompt_sizes:
                    try:
                        total_size = float(result.get('total_size', 0.0))
                    except Exception:
                        total_size = 0.0
                    # keep the same scaling as elsewhere; apply_hyper may overwrite this divisor ("/ 6")
                    prompt_sizes[llm_anchor] = total_size / 6
            # output in a fixed key order to guarantee 4 dims per node
            w_on_ip = [prompt_sizes["1"], prompt_sizes["2"], prompt_sizes["3"], prompt_sizes["4"]]
            workloads.append(w_on_ip)
        workloads = [item for sublist in workloads for item in sublist]

        llm_runing_status = []
        for node in self.nodes:
            pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node}')
            pod_status = [0] * 4
            for pod in pods.items:
                if pod.metadata.labels:
                    for i in range(1, 5):
                        pod_label = f"llm_{i}"
                        if pod.metadata.labels.get('llm') == pod_label:
                            pod_status[i - 1] = 1
            llm_runing_status.append(pod_status)
        llm_runing_status = [item for sublist in llm_runing_status for item in sublist]

        self.global_state = requests + resources + workloads + llm_runing_status
        # Print one-time diagnostic info to verify per-node segment lengths.
        if not getattr(self, "_gs_debug_printed", False):
            try:
                per_node_res_len = len(self.current_resource[0]) if self.current_resource else 0
                print(
                    f"[global_state diagnostic] nodes={len(self.nodes)}; per-node lens: "
                    f"requests=4, resources={per_node_res_len}, workloads=4, status=4; "
                    f"totals: requests={len(requests)}, resources={len(resources)}, "
                    f"workloads={len(workloads)}, status={len(llm_runing_status)}, "
                    f"global={len(self.global_state)}"
                )
            except Exception:
                pass
            self._gs_debug_printed = True
        return self.global_state


    async def generate_actor_state(self):
        self.actor_states = []
        
        for node, node_ip, request, local_resource in zip(self.nodes, self.node_ips, self.request_list, self.current_resource):
            normalized_local_request = [value / r[1] for value, r in zip(request, self.request_range)]

            pipeline = [
                {
                    "$group": {
                        "_id": "$metadata.llm_anchor", 
                        "total_size": {"$sum": {"$toDouble": "$metadata.size"}}
                    }
                }
            ]
            prompt_sizes = {"1": 0, "2": 0, "3": 0, "4": 0}
            prompt_col = self.mongo["prompt_recorder"][f"{node_ip.replace('.', '_')}"]
            results = prompt_col.aggregate(pipeline)

            async for result in results:
                llm_anchor = str(result['_id'])
                if llm_anchor in prompt_sizes:
                    total_size = result['total_size']
                    prompt_sizes[llm_anchor] = total_size / 6
                else:
                    pass

            w_on_ip = list(prompt_sizes.values())

            pods = self.v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node}')
            loacl_pod_status = [0] * 4
            for pod in pods.items:
                if pod.metadata.labels:
                    for i in range(1, 5):
                        pod_label = f"llm_{i}"
                        if pod.metadata.labels.get('llm') == pod_label:
                            loacl_pod_status[i - 1] = 1

            actor_state = normalized_local_request + local_resource + w_on_ip + loacl_pod_status

            # print(f"[Debug] node={node_ip}, actor_state length={len(actor_state)}, actor_state={actor_state}")

            self.actor_states.append(actor_state)

        return self.actor_states


    def _phase_request_range(self, slot):
        # When WORKLOAD_PROFILE=stress3phase, the per-LM arrival range is a
        # function of the slot (see _STRESS3PHASE). Otherwise the constant
        # request_range passed at construction is used (unchanged behaviour).
        if os.getenv("WORKLOAD_PROFILE", "").strip() == "stress3phase":
            for lo, hi, rng in _STRESS3PHASE:
                if lo <= slot < hi:
                    return rng
            return _STRESS3PHASE[-1][2]  # clamp past end of schedule
        return self.request_range

    async def generate_slot_llm_map(self):
        self.slot_llm_map = {}
        # WORKLOAD_SEED gives each (seed, slot, node) a deterministic per-LM
        # arrival count for reproducible workload generation.
        ws_env = os.getenv("WORKLOAD_SEED", "").strip()
        if ws_env:
            try:
                ws = int(ws_env)
            except Exception:
                ws = 0
            slot = int(getattr(self, "slot_anchor", 0))
            rr = self._phase_request_range(slot)
            for i, ip in enumerate(self.node_ips):
                counts = []
                for k in range(4):
                    # integer seed -> unambiguous, reproducible; independent
                    # stream per (workload_seed, slot, node, lm).
                    seed = ws * 1000003 + slot * 1009 + i * 97 + k * 17
                    rng = random.Random(seed)
                    counts.append(rng.randint(rr[k][0], rr[k][1]))
                self.slot_llm_map[ip] = counts
        else:
            for i, ip in enumerate(self.node_ips):
                llm1 = random.randint(self.request_range[0][0], self.request_range[0][1])
                # Anchor 2 is enabled again: generate workload similar to other anchors.
                llm2 = random.randint(self.request_range[1][0], self.request_range[1][1])
                llm3 = random.randint(self.request_range[2][0], self.request_range[2][1])
                llm4 = random.randint(self.request_range[3][0], self.request_range[3][1])
                self.slot_llm_map[ip] = [llm1, llm2, llm3, llm4]

        # Persist the TRUE per-LM offered load for this slot so the macro
        # planner can observe the actual workload volume. (The latency-cell
        # count saturates at num_slots*num_nodes and cannot represent volume.)
        # Best-effort: telemetry must never break workload generation.
        try:
            slot_idx = int(getattr(self, "slot_anchor", 0))
            totals = [0, 0, 0, 0]
            for counts in self.slot_llm_map.values():
                for k in range(4):
                    totals[k] += int(counts[k])
            await self.mongo["workload"][self.dbname].update_one(
                {"slot_anchor": int(slot_idx)},
                {"$set": {
                    "slot_anchor": int(slot_idx),
                    "per_lm": {"1": totals[0], "2": totals[1],
                               "3": totals[2], "4": totals[3]},
                }},
                upsert=True,
            )
        except Exception:
            pass

    async def generate_requests_async(self):
        self.request_list = []
        for ip in self.node_ips:
            self.request_list.append(self.slot_llm_map[ip])

        async def send_generate_request(i):
            nodeip = str(self.node_ips[i])
            url = f"http://{nodeip}:30000/generate"
            llm1, llm2, llm3, llm4 = self.request_list[i] 
            payload = {
                "llm1": llm1,
                "llm2": llm2,
                "llm3": llm3,
                "llm4": llm4,
                "slot_anchor": self.slot_anchor
            }
            response = await self.post_with_retries_async(url, json=payload, timeout=10)
            if response and response.json().get("status") == "success":
                print(f"Request to {nodeip} successful.")
                return True
            else:
                print(f"Request to {nodeip} ultimately failed after retries.")
                return False

        results = await asyncio.gather(*[send_generate_request(i) for i in range(len(self.node_ips))])

        if all(results):
            print("All generate requests completed successfully.")
        else:
            print("Some generate requests failed. Handle accordingly.")

    async def wait_until_all_files_ready(self, max_wait=30, regenerate_interval=5):
        start = time.time()
        not_ready_nodes = set(self.node_ips)

        while True:
            still_not_ready = []
            for ip in not_ready_nodes:
                url = f"http://{ip}:30000/check_files_ready"
                payload = {"slot_anchor": self.slot_anchor}
                resp = await self.post_with_retries_async(url, json=payload)
                status = resp.json().get("status") if (resp and resp.json()) else "not_ready"
                if status != "ready":
                    still_not_ready.append(ip)

            if not still_not_ready:
                print("All nodes files are ready.")
                return True

            elapsed = time.time() - start
            if elapsed > max_wait:
                print(f"Timeout waiting for files to be ready. Not ready nodes: {still_not_ready}")

                for ip in still_not_ready:
                    cleanup_url = f"http://{ip}:30000/cleanup_slot"
                    cleanup_payload = {"slot_anchor": self.slot_anchor}
                    cleanup_resp = await self.post_with_retries_async(cleanup_url, json=cleanup_payload, timeout=10)
                    if cleanup_resp and cleanup_resp.json().get("status") == "success":
                        print(f"Cleanup slot {self.slot_anchor} on {ip} success.")
                    else:
                        print(f"Cleanup slot {self.slot_anchor} on {ip} failed or not confirmed.")
                    
                    idx = self.node_ips.index(ip)
                    llm1, llm2, llm3, llm4 = self.request_list[idx]
                    generate_url = f"http://{ip}:30000/generate"
                    gen_payload = {
                        "llm1": llm1,
                        "llm2": llm2,
                        "llm3": llm3,
                        "llm4": llm4,
                        "slot_anchor": self.slot_anchor
                    }
                    gen_resp = await self.post_with_retries_async(generate_url, json=gen_payload, timeout=10)
                    if gen_resp and gen_resp.json().get("status") == "success":
                        print(f"Regenerate request to {ip} successful.")
                    else:
                        print(f"Regenerate request to {ip} ultimately failed after retries.")

                start = time.time()
                not_ready_nodes = set(still_not_ready)
            else:
                await asyncio.sleep(regenerate_interval)

    async def process_single_node(self, node_ip, slot_anchor, destinations):
        url = f"http://{node_ip}:30000/process_data"
        payload = {
            "destinations": destinations,
            "slot_anchor": slot_anchor
        }
        resp = await self.post_with_retries_async(url, json=payload, timeout=10)
        return node_ip, resp

    async def process_data_on_all_nodes_concurrently(self, slot_anchor, global_destinations, max_wait=30, regenerate_interval=2):
        start = time.time()
        while True:
            tasks = []
            for node_idx, node_ip in enumerate(self.node_ips):
                tasks.append(asyncio.create_task(self.process_single_node(node_ip, slot_anchor, global_destinations[node_idx])))

            done = await asyncio.gather(*tasks, return_exceptions=True)

            all_success = True
            for (node_ip, resp) in done:
                if resp is None:
                    print(f"Request to {node_ip} failed after all retries. Retrying...")
                    all_success = False
                    break
                else:
                    resp_json = resp.json()
                    if resp_json.get("status") != "success":
                        print(f"Request to {node_ip} returned non-success status: {resp_json}. Retrying...")
                        all_success = False
                        break

            if all_success:
                print("All nodes returned success. Proceeding with next steps...")
                return True
            else:
                elapsed = time.time() - start
                if elapsed > max_wait:
                    raise Exception(f"Exceeded maximum wait time of {max_wait} seconds without all nodes succeeding.")
                print(f"Not all nodes succeeded yet. Waiting {regenerate_interval}s before retrying...")
                await asyncio.sleep(regenerate_interval)

    async def wait_until_db_ready(self, max_wait=30):
        start = time.time()
        while True:
            all_ready = True
            for ip in self.node_ips:
                url = f"http://{ip}:30000/check_db_ready"
                resp = await self.post_with_retries_async(url, json={}, timeout=10)
                if not (resp and resp.json().get("status") == "ready"):
                    all_ready = False
                    break
            if all_ready:
                print("All nodes db are ready.")
                return True
            if time.time() - start > max_wait:
                print("Timeout waiting for db to be ready.")
                return False
            await asyncio.sleep(1)

    async def trigger_all_nodes_async(self, slot_anchor, status, incre_from):
        tasks = []
        for node_ip in self.node_ips:
            trigger_url = f"http://{node_ip}:30000/trigger"
            payload_1 = {
                "slot_anchor": slot_anchor,
                "status": status,
                "incre_from": incre_from
            }
            tasks.append((node_ip, trigger_url, payload_1))

        results = await asyncio.gather(*[
            self.post_with_retries_async(url, json=payload) for (node_ip, url, payload) in tasks
        ])

        success_count = 0
        for (node_info, resp) in zip(tasks, results):
            node_ip, _, _ = node_info
            if resp and resp.status_code == 200:
                print(f"trigger Success for {node_ip}:", resp.json())
                success_count += 1
            else:
                print(f"trigger Error for {node_ip}")

        if success_count == len(tasks):
            print("All triggers completed successfully, proceed with next steps...")
        else:
            print("Some triggers failed, handle accordingly.")

    async def set_dbname_on_all_nodes(self, dbname, observe, explore, memory_length, batch_size, length_of_slot):
        success_count = 0
        for ip in self.node_ips:
            url = f"http://{ip}:30000/get_dbname"
            payload = {
                "dbname": dbname,
                "observe": observe,
                "explore": explore,
                "memory_length": memory_length,
                "batch_size": batch_size,
                "length_of_slot": length_of_slot
            }
            response = await self.post_with_retries_async(url, json=payload, timeout=10)
            if response and response.json().get("status") == "success":
                success_count += 1
            else:
                print(f"Failed to set db config on {ip}")
        return success_count == len(self.node_ips)

    async def set_agent_mode_on_all_nodes(self, mode: str):
        success_count = 0
        for ip in self.node_ips:
            url = f"http://{ip}:30000/set_agent_mode"
            payload = {"mode": mode}
            response = await self.post_with_retries_async(url, json=payload, timeout=10)
            if response and response.json().get("status") == "success":
                success_count += 1
            else:
                print(f"Failed to set agent mode on {ip}")
        return success_count == len(self.node_ips)

    async def set_node_layout_intent_on_all_nodes(self, layout_by_node: dict):
        """Propagate per-node layout intent (macro role) to all worker nodes.

        layout_by_node is expected to be keyed by node hostname (e.g. worker1),
        with values being arbitrary JSON-serialisable dicts describing this
        node's role (e.g. which anchors to prioritise on CPU/GPU).
        """
        success_count = 0
        for hostname, ip in zip(self.nodes, self.node_ips):
            # Prefer hostname keys; fall back to IP keys if provided that way.
            role = layout_by_node.get(hostname) or layout_by_node.get(ip) or {}
            url = f"http://{ip}:30000/set_node_layout_intent"
            payload = {"layout": role}
            response = await self.post_with_retries_async(url, json=payload, timeout=10)
            if response and response.json().get("status") == "success":
                success_count += 1
            else:
                print(f"[NODE-ROLE] Failed to set layout intent on {hostname} ({ip})")
        return success_count == len(self.node_ips)

    async def reward_retrieval(self):
        """Aggregate global reward directly from per-node local collections.
        Barrier semantics: only emit a slot when EVERY node has a processed=True
        entry for that local_slot_anchor. Uses a consumed collection to avoid
        reusing the same slot multiple times.
        """
        # Ensure unique index for consumed markers
        try:
            await self.consumed_col.create_index([("slot_anchor", 1)], unique=True)
        except Exception:
            pass

        # Build per-node maps: {slot(int): reward(float)} for processed entries
        per_node_maps = {}
        slot_sets = []
        for ip in self.node_ips:
            ipu = ip.replace('.', '_')
            col = self.reward_db[ipu]
            # projection limits transfer size
            cursor = col.find({"processed": True}, {"_id": 0, "local_slot_anchor": 1, "reward": 1})
            docs = await cursor.to_list(length=None)
            slot_to_reward = {}
            slots = set()
            for d in docs:
                lsa = d.get("local_slot_anchor")
                try:
                    lsa_int = int(lsa)
                except Exception:
                    continue
                try:
                    r = float(d.get("reward", 0.0))
                except Exception:
                    r = 0.0
                slot_to_reward[lsa_int] = r
                slots.add(lsa_int)
            per_node_maps[ipu] = slot_to_reward
            slot_sets.append(slots)

        if not slot_sets:
            return []

        # Slots present on ALL nodes (barrier)
        ready_all = set.intersection(*slot_sets) if len(slot_sets) > 1 else slot_sets[0]

        # Exclude already-consumed slots
        consumed_docs = await self.consumed_col.find({}, {"_id": 0, "slot_anchor": 1}).to_list(length=None)
        consumed = set()
        for cd in consumed_docs:
            try:
                consumed.add(int(cd.get("slot_anchor")))
            except Exception:
                continue
        pending = sorted(ready_all - consumed)

        retrievaled_reward = []
        # Optionally limit batch size per call to avoid long loops
        for sa in pending[:200]:
            total = 0.0
            ok = True
            for ip in self.node_ips:
                ipu = ip.replace('.', '_')
                m = per_node_maps.get(ipu, {})
                if sa not in m:
                    ok = False
                    break
                total += float(m[sa])
            if not ok:
                continue
            avg = total / float(len(self.node_ips)) if self.node_ips else 0.0
            retrievaled_reward.append([sa, avg])
            try:
                await self.consumed_col.insert_one({
                    "slot_anchor": sa,
                    "reward": avg,
                    "ts": time.time()
                })
            except Exception:
                # ignore duplicate key or transient errors
                pass

        return retrievaled_reward

    async def _cleanup_ma_reward_duplicates(self):
        """Remove any duplicate docs in ma_reward that share the same slot_anchor.
        Keep at most one document per slot_anchor, preferring the one whose _id equals slot_anchor.
        """
        pipeline = [
            {"$group": {"_id": "$slot_anchor", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"_id": {"$ne": None}, "count": {"$gt": 1}}}
        ]
        cursor = self.reward_col.aggregate(pipeline)
        async for grp in cursor:
            slot_key = grp.get("_id")
            id_list = grp.get("ids", [])
            # prefer the doc whose _id equals slot_anchor
            keep_id = None
            for _id in id_list:
                if str(_id) == str(slot_key):
                    keep_id = _id
                    break
            if keep_id is None and id_list:
                keep_id = id_list[0]
            # delete all other docs for this slot_anchor
            await self.reward_col.delete_many({"slot_anchor": slot_key, "_id": {"$ne": keep_id}})
        # After cleanup, enforce uniqueness on slot_anchor for future safety (partial to allow missing field)
        try:
            await self.reward_col.create_index(
                [("slot_anchor", 1)], unique=True,
                partialFilterExpression={"slot_anchor": {"$exists": True}}
            )
        except Exception:
            pass

    async def reset(self):
        for ip in self.node_ips:
            url = f"http://{ip}:30000/cleanup_prompt"
            await self.post_with_retries_async(url, json={}, timeout=10)

        pattern = re.compile(r"^llm(\d+)-worker(\d+)$")
        all_pods = self.v1.list_pod_for_all_namespaces().items
        for pod in all_pods:
            pod_name = pod.metadata.name
            namespace = pod.metadata.namespace
            if pattern.match(pod_name):
                self.v1.delete_namespaced_pod(name=pod_name, namespace=namespace, grace_period_seconds=0)
                print(f"Pod {pod_name} in namespace {namespace} deleted.")

        await self.generate_slot_llm_map()

        # For training environments (e.g. 'sa', 'ma') we eagerly generate
        # the first batch of requests during reset so that the initial
        # global/actor state includes real workload. For pure routing
        # mode ("routing"), we rely on step() to generate requests so
        # that each slot (including slot 0) has exactly one generate
        # cycle; this avoids issuing duplicate /generate commands on
        # slot_anchor=0.
        if self.status != "routing":
            await self.generate_requests_async()
            ready = await self.wait_until_all_files_ready()
            if not ready:
                print(f"Step {self.slot_anchor}: Not all nodes ready within time limit.")
            else:
                print(f"Step {self.slot_anchor}: All files are ready on all nodes.")

            global_state = await self.generate_global_state()
            actor_state = await self.generate_actor_state()
            return global_state, actor_state

        # In routing mode, reset only clears queues/pods and initialises
        # slot_llm_map; step() will handle request generation.
        return None, None

    async def reset_prompt_queues(self):
        """Clear per-node prompt/output queues without deleting LLM pods."""
        for ip in self.node_ips:
            url = f"http://{ip}:30000/cleanup_prompt"
            await self.post_with_retries_async(url, json={}, timeout=10)

    async def step(self, actions):
        global_state = await self.generate_global_state()
        actor_state = await self.generate_actor_state()

        # Build candidate lists per head: text heads (1,2) -> all nodes; image heads (3,4) -> GPU nodes
        gpu_node_indices = [i for i, n in enumerate(self.nodes) if self.node_resources.get(n, {}).get('max_gpu', False)]
        all_node_indices = list(range(len(self.node_ips)))

        # Fallback: if no GPU nodes, still allow mapping to self (index 0) to keep shapes consistent
        if len(gpu_node_indices) == 0:
            gpu_node_indices = [0]

        head_candidates = [
            all_node_indices,  # head 0 for anchor 1
            all_node_indices,  # head 1 for anchor 2
            gpu_node_indices,  # head 2 for anchor 3 (image)
            gpu_node_indices   # head 3 for anchor 4 (image)
        ]

        global_destinations = []
        for action in actions:
            destination = []
            for head_idx, prompt_to in enumerate(action):
                candidates = head_candidates[head_idx] if head_idx < len(head_candidates) else all_node_indices
                # prompt_to is an integer index within candidates (since actor outputs categorical over action_shapes[head])
                local_idx = int(prompt_to)
                local_idx = max(0, min(local_idx, len(candidates) - 1))
                node_idx = candidates[local_idx]
                destination.append(self.node_ips[node_idx])
            global_destinations.append(destination)

        # 1. Generate requests
        await self.generate_requests_async()
        # 2. Wait until files ready
        ready = await self.wait_until_all_files_ready()
        if not ready:
            print(f"Step {self.slot_anchor}: Not all nodes ready within time limit.")
        else:
            print(f"Step {self.slot_anchor}: All files are ready on all nodes.")

        # 4. Process data with matd3 made destinations
        all_success = await self.process_data_on_all_nodes_concurrently(self.slot_anchor, global_destinations)

        # Advance the on-node controller (Lyapunov or GPTLocal) once per
        # logical slot via /trigger. This keeps the
        # local agent active even when routing is externally supplied so that the
        # lower layer can react to load and manage pod layouts.
        if all_success and self.slot_anchor % self.length_of_slot == 0:
            await self.trigger_all_nodes_async(self.slot_anchor, self.status, self.incre_from)

        self.slot_anchor += 1
        await self.generate_slot_llm_map()
        next_global_state = await self.generate_global_state()

        return actor_state, global_state, next_global_state

    async def get_last_anchor(self):
        from pymongo.collation import Collation
        NUMERIC = Collation(locale="en", numericOrdering=True)
        self.latency_col_node1 = self.mongo["latency"][f"{self.node_ips[0].replace('.', '_')}"]
        cursor = (
            self.latency_col_node1.find({}, {"slot_anchor": 1, "_id": 0})
            .sort("slot_anchor", -1)
            .collation(NUMERIC)
            .limit(1)
        )
        docs = await cursor.to_list(length=1)
        if docs:
            self.slot_anchor = int(docs[0]["slot_anchor"]) + 1
            self.incre_from = self.slot_anchor
            return
        else:
            return
