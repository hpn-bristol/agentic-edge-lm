import datetime
import os
import time
from kubernetes import client, config
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient

class Monitor:
    def __init__(self):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self.v1 = client.CoreV1Api()
        self.appsv1 = client.AppsV1Api()
        self.node_metrics = {}
        self.prev_metrics = {}
        self.nodes = self.v1.list_node().items
        self.node_ips, self.jetson_ips = self.fetch_ips(gpu=True, jetson = False)
        self.metrics_recorder = []
        service = self.v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        self.mongo = MongoClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")
        # If the orchestrator (e.g. policy_manager.main) already chose a run
        # dbname, share it so the monitor's k8smonitor.<dbname> aligns with
        # latency.<...> / prompt_recorder.<...> for the same run.
        env_db = (os.getenv("LLM_ROUTING_DBNAME") or "").strip()
        self.dbname = env_db if env_db else f"run-{time.time()}"
        self.col = self.mongo["k8smonitor"][self.dbname]
        self.parsed = {
            'free_memory': 0,
            'total_memory': 0,
            'tx_bytes': 0,
            'rx_bytes': 0,
            'tx_rate': 0,
            'rx_rate': 0,
            'idle_cpu_seconds': 0,
            'total_cpu_seconds': 0,
            'gpu_freq':0,
            'vram_freq':0,
            'enc_util':0,
            'dec_util':0,
            'power_usage':0,
            'gpu_util':0,
            'vram_util':0,
            'vram_free':0,
            'vram_used':0
        }
        self.metric_lookup = {
            'node_memory_MemAvailable_bytes': 'free_memory',
            'node_memory_MemTotal_bytes': 'total_memory',
            'node_network_transmit_bytes_total': 'tx_bytes',
            'node_network_receive_bytes_total': 'rx_bytes',
            # 'node_cpu_seconds_total': 'total_cpu_seconds',
            'DCGM_FI_DEV_SM_CLOCK': 'gpu_freq',
            'DCGM_FI_DEV_MEM_CLOCK': 'vram_freq',
            'DCGM_FI_DEV_ENC_UTIL': 'enc_util',
            'DCGM_FI_DEV_DEC_UTIL': 'dec_util',
            'DCGM_FI_DEV_POWER_USAGE': 'power_usage',
            'DCGM_FI_DEV_GPU_UTIL': 'gpu_util',
            'DCGM_FI_DEV_MEM_COPY_UTIL': 'vram_util',
            'DCGM_FI_DEV_FB_FREE': 'vram_free',
            'DCGM_FI_DEV_FB_USED': 'vram_used',
        }

    def fetch_ips(self, gpu=False, jetson = False):
        node_ips = {node.metadata.name: {"node_ip":node.status.addresses[0].address} for node in self.nodes}
        jetson_ips = {}

        if gpu:
            daemonset = self.appsv1.read_namespaced_daemon_set(name="nvidia-dcgm-exporter", namespace="gpu-operator")
            label_selector = daemonset.spec.selector.match_labels
            label_selector = ",".join([f"{key}={value}" for key, value in label_selector.items()])
            pods = self.v1.list_namespaced_pod(namespace="nvidia", label_selector=label_selector)
            for pod in pods.items:
                node_name = pod.spec.node_name
                pod_ip = pod.status.pod_ip
                node_ips[node_name]["gpu_ip"] = pod_ip

        if jetson:
            daemonset = self.appsv1.read_namespaced_daemon_set(name="jetson-stats-exporter", namespace="monitoring")
            label_selector = daemonset.spec.selector.match_labels
            label_selector = ",".join([f"{key}={value}" for key, value in label_selector.items()])
            pods = self.v1.list_namespaced_pod(namespace="monitoring", label_selector=label_selector)
            for pod in pods.items:
                node_name = pod.spec.node_name
                pod_ip = pod.status.pod_ip
                jetson_ips[node_name] = pod_ip

        return node_ips, jetson_ips
    
    def fetch_metrics(self, node_ip=None, gpu=False, jetson=False, gpu_ip=None, jetson_ip=None):
        if not jetson:
            try:
                response = requests.get(f"http://{node_ip}:9100/metrics")
                response.raise_for_status()
                metrics = response.text
            except requests.exceptions.RequestException as e:
                print(f"Error fetching metrics from {node_ip}: {e}")
                return None
        
            if gpu:
                try:
                    response = requests.get(f"http://{gpu_ip}:9400/metrics")
                    response.raise_for_status()
                    metrics += '/n' + response.text
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching metrics from {gpu_ip}: {e}")
                    return None
            
        elif jetson:
            try:
                response = requests.get(f"http://{jetson_ip}:9400")
                response.raise_for_status()
                metrics = response.text
            except requests.exceptions.RequestException as e:
                print(f"Error fetching metrics from {jetson_ip}: {e}")
                return None
            
        return metrics

    def parse_metrics(self, metrics, prev_metrics, interval):
        self.parsed = {key: 0 for key in self.parsed}

        for line in metrics.split('\n'):
            # Skip comment and help lines
            if line.startswith('#') or line.strip() == "":
                continue

            if '{' in line or '} ' in line:
                try:
                    key, value = line.split('} ', 1)  # Only split once
                    key = key.split("{")[0]
                    value = float(value)
                except ValueError as v:
                    print(f"ValueError during parsing line with {v}, line: '{line}'")
                    continue
            elif ' ' in line:  
                try:
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        print(f"Skipping line with insufficient parts after split by space: '{line}'")
                        continue
                    key, value = parts
                    value = float(value)
                except ValueError as v:
                    print(f"ValueError during parsing line with space: {v}, line: '{line}'")
                    continue
            else:
                # print(f"Skipping unexpected format line: '{line}'")
                continue

            if key in self.metric_lookup:
                self.parsed[self.metric_lookup[key]] = value

            # Handle idle CPU (special case)
            # elif key == 'node_cpu_seconds_total' and 'mode="idle"' in line:
            #     self.parsed['idle_cpu_seconds'] += value
            
            elif key == 'node_cpu_seconds_total':
                if 'mode="idle"' in line:
                    self.parsed['idle_cpu_seconds'] += value
                # Ensure 'total_cpu_seconds' is initialized before adding to it
                self.parsed['total_cpu_seconds'] += value

            #TODO Jetson stuff
            
        
        # Calculate differences and convert to required units if previous metrics exist
        if prev_metrics:
            cpu_delta = self.parsed['idle_cpu_seconds'] - prev_metrics['idle_cpu_seconds']
            total_cpu_delta = self.parsed['total_cpu_seconds'] - prev_metrics['total_cpu_seconds']
            if total_cpu_delta > 0:  # Avoid division by zero
                self.parsed['cpu'] = 100.0 * (1 - cpu_delta / total_cpu_delta)
            
            # Use a constant for the calculation factor
            rate_factor = 8 / interval / 1e6  
            self.parsed['tx_rate'] = (self.parsed['tx_bytes'] - prev_metrics['tx_bytes']) * rate_factor
            self.parsed['rx_rate'] = (self.parsed['rx_bytes'] - prev_metrics['rx_bytes']) * rate_factor
        
        return self.parsed

    # TODO needs a rethink
    def collect_metrics(self, duration_seconds, interval, livesave=False):
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            with ThreadPoolExecutor(max_workers=len(self.node_ips)+len(self.jetson_ips)) as executor: 
                future_to_ip = {executor.submit(self.fetch_metrics, node_ip = ip["node_ip"], gpu = False): name for name, ip in self.node_ips.items() if "gpu_ip" not in ip.keys()} # Non-gpu nodes
                future_to_ip.update({executor.submit(self.fetch_metrics, node_ip = ip["node_ip"], gpu_ip = ip["gpu_ip"], gpu = True): name for name, ip in self.node_ips.items() if "gpu_ip" in ip.keys()})
                current_metrics = {}
                for future in as_completed(future_to_ip):
                    node_name = future_to_ip[future]
                    data = future.result()
                    if data:  
                        if node_name in self.prev_metrics:
                            parsed_metrics = self.parse_metrics(data, self.prev_metrics[node_name], interval)
                        else:
                            parsed_metrics = self.parse_metrics(data, {}, interval)
                        current_metrics[node_name] = parsed_metrics
                self.prev_metrics = current_metrics
                self.node_metrics = current_metrics
                # if time.time()-start_time > interval:   #if data over time interval can be recorded, start recording. 
                #     self.metrics_recorder.append(self.node_metrics)
                #     self.metrics_recorder=self.metrics_recorder[1:] #delete data thats past the time interval
                #     self.prev_metrics=self.metrics_recorder[0] #replace immediate prev_metrics with past metrics (interval)
                #     # print("PAST TIME: ",self.metrics_recorder[0])
                # else:
                #     self.metrics_recorder.append(self.node_metrics)
                if livesave:
                    self.save_metrics_as_json()  # save
            time.sleep(interval)

    def return_metrics_as_dict(self):
        return {"timestamp": datetime.datetime.now(), "metrics": self.node_metrics}

    def monitor_process(self, duration_seconds, interval):
        self.collect_metrics(duration_seconds=duration_seconds, interval=interval, livesave=True)

    def save_metrics_as_json(self, output_file="data.json"):
        # print("inserting ")
        if self.node_metrics:
            self.col.insert_one({"timestamp": datetime.datetime.now(), "metrics": self.node_metrics})
            # with open(output_file, 'w') as f:
            #     json.dump(self.node_metrics, f, indent=4)
    
    def collect_reward(self):
        results = []
        for collection_name in self.mongo["latency"].list_collection_names():
            col = self.mongo["latency"][collection_name]
            ip = collection_name.replace('latency_recorder_', '')
            documents = col.find()

            for doc in documents:
                slot_anchor = doc["slot_anchor"]
                
                delays = [
                    doc.get(slot_anchor, {}).get('1', {}).get('delay', None),
                    doc.get(slot_anchor, {}).get('2', {}).get('delay', None),
                    doc.get(slot_anchor, {}).get('3', {}).get('delay', None),
                    doc.get(slot_anchor, {}).get('4', {}).get('delay', None)
                ]
                
                if all(d is not None for d in delays):
                    average_delay = sum(delays) / len(delays)
                    
                    # col.update_one(
                    #     {"_id": doc["_id"]},
                    #     {"$set": {"overall_delay": average_delay}}
                    # )

                    col.delete_one({"_id": doc["_id"]})
                    
                    results.append([collection_name, slot_anchor, average_delay]) # this is for central use

                    self.mongo["reward"][ip].insert_one({ # this is for local use.
                        'slot_anchor': slot_anchor,
                        'average_delay': average_delay
                    })

        print(results)


if __name__ == "__main__":
    # Standalone monitor entry point, spawned by policy_manager.main() as a
    # background process. dbname is read from LLM_ROUTING_DBNAME so that the
    # k8smonitor.<dbname> collection aligns with the controller's run.
    duration = int(os.getenv("MONITOR_DURATION_SEC", "864000"))  # 10 days default
    interval = int(os.getenv("MONITOR_INTERVAL_SEC", "3"))
    m = Monitor()
    print(f"[monitor] started: dbname={m.dbname} duration={duration}s interval={interval}s")
    try:
        m.collect_metrics(duration_seconds=duration, interval=interval, livesave=True)
    except KeyboardInterrupt:
        print("[monitor] caught Ctrl+C, exiting cleanly.")



