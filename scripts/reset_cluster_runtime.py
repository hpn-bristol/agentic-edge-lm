#!/usr/bin/env python3
"""Reset cluster-side runtime state before launching an MA run.

This utility clears cluster-side runtime state before a new MA run:

1. Drop non-system MongoDB databases used by previous runs.
2. Ask each worker-side FastAPI service to clear local runtime state via
   ``POST /cleanup``.

Worker nodes are discovered from the Kubernetes node list. Control-plane nodes
are skipped. The script expects the worker service to be reachable through the
host-network port used by this project.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Iterable, List

import requests
from kubernetes import client, config
from pymongo import MongoClient


SYSTEM_DBS = {"admin", "local", "config"}


def natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def load_kube() -> None:
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()


def discover_worker_ips() -> List[str]:
    v1 = client.CoreV1Api()
    workers = []
    for node in sorted(v1.list_node().items, key=lambda n: natural_sort_key(n.metadata.name or "")):
        labels = node.metadata.labels or {}
        if labels.get("node-role.kubernetes.io/controlplane"):
            continue
        if labels.get("dedicated") == "controller":
            continue
        ips = [
            addr.address
            for addr in (node.status.addresses or [])
            if addr.type == "InternalIP" and addr.address
        ]
        if ips:
            workers.append(ips[0])
    return workers


def mongodb_uri() -> str:
    v1 = client.CoreV1Api()
    svc = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
    return f"mongodb://root:secret@{svc.spec.cluster_ip}:27017"


def drop_runtime_databases(uri: str, dry_run: bool = False) -> None:
    cli = MongoClient(uri)
    for db_name in cli.list_database_names():
        if db_name in SYSTEM_DBS:
            continue
        print(f"[reset-cluster] drop database: {db_name}")
        if not dry_run:
            cli.drop_database(db_name)


def post_cleanup(ip: str, port: int, retries: int, timeout: float, retry_delay: float, dry_run: bool) -> bool:
    url = f"http://{ip}:{port}/cleanup"
    if dry_run:
        print(f"[reset-cluster] dry-run cleanup: {url}")
        return True
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json={}, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json() if resp.content else {}
                if payload.get("status") in (None, "success"):
                    print(f"[reset-cluster] cleanup ok: {ip}")
                    return True
            print(f"[reset-cluster] cleanup non-success {ip}: status={resp.status_code}, body={resp.text[:200]}")
        except Exception as exc:
            print(f"[reset-cluster] cleanup attempt {attempt}/{retries} failed for {ip}: {exc}")
        time.sleep(retry_delay)
    return False


def cleanup_workers(
    ips: Iterable[str],
    port: int,
    retries: int,
    timeout: float,
    retry_delay: float,
    dry_run: bool,
) -> None:
    failed = []
    for ip in ips:
        if not post_cleanup(ip, port, retries, timeout, retry_delay, dry_run):
            failed.append(ip)
    if failed:
        raise RuntimeError(f"worker cleanup failed for: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset MA runtime state in Kubernetes and MongoDB.")
    parser.add_argument("--worker-port", type=int, default=30000)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_kube()
    workers = discover_worker_ips()
    if not workers:
        sys.exit("[reset-cluster] no worker nodes discovered")

    print(f"[reset-cluster] workers={workers}")
    drop_runtime_databases(mongodb_uri(), dry_run=args.dry_run)
    cleanup_workers(
        workers,
        port=args.worker_port,
        retries=args.retries,
        timeout=args.timeout,
        retry_delay=args.retry_delay,
        dry_run=args.dry_run,
    )
    print("[reset-cluster] done")


if __name__ == "__main__":
    main()
