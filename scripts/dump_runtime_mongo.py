#!/usr/bin/env python3
"""Dump one MA run's MongoDB state to a directory, JSON per collection.

Usage:
  dump_runtime_mongo.py <run_dbname> <out_dir>

run_dbname is the run-scoped database name used to
locate per-run agent_events and workload collections). Always-dumped databases
are: latency, prompt_recorder, k8smonitor, reward.
"""
from __future__ import annotations

import json
import pathlib
import sys

from kubernetes import client, config
from pymongo import MongoClient


FIXED_DBS = ("latency", "prompt_recorder", "k8smonitor", "reward")
TAGGED_DBS = ("agent_events", "workload")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: dump_runtime_mongo.py <run_dbname> <out_dir>")
    run_dbname = sys.argv[1].strip()
    out_dir = pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    svc = client.CoreV1Api().read_namespaced_service(
        name="mongodb-service", namespace="mongodb"
    )
    cli = MongoClient(f"mongodb://root:secret@{svc.spec.cluster_ip}:27017")

    total_docs = 0
    for db_name in FIXED_DBS:
        db = cli[db_name]
        for col in db.list_collection_names():
            docs = list(db[col].find({}))
            for d in docs:
                d.pop("_id", None)
            sub = out_dir / db_name
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"{col}.json").write_text(
                json.dumps(docs, ensure_ascii=False, default=str)
            )
            print(f"  {db_name}.{col}: {len(docs)} docs")
            total_docs += len(docs)

    if run_dbname:
        for db_name in TAGGED_DBS:
            try:
                col = cli[db_name][run_dbname]
                docs = list(col.find({}))
                for d in docs:
                    d.pop("_id", None)
                sub = out_dir / db_name
                sub.mkdir(parents=True, exist_ok=True)
                (sub / f"{run_dbname}.json").write_text(
                    json.dumps(docs, ensure_ascii=False, default=str)
                )
                print(f"  {db_name}.{run_dbname}: {len(docs)} docs")
                total_docs += len(docs)
            except Exception as exc:
                print(f"  warn {db_name}.{run_dbname}: {exc}")

    print(f"dump done -> {out_dir}, total {total_docs} docs")


if __name__ == "__main__":
    main()
