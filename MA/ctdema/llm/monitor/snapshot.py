"""Telemetry snapshot utilities for the LLM routing controller."""

from __future__ import annotations

import datetime as _dt
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pymongo import MongoClient
from kubernetes import client as k8s_client, config as k8s_config


DEFAULT_MONGO_URI = os.getenv("LLM_MONGO_URI")


class SnapshotBuilder:
    """Aggregate latency and utilisation metrics over a rolling window."""

    def __init__(
        self,
        mongo_uri: Optional[str] = DEFAULT_MONGO_URI,
        window_slots: int = 30,
        latency_db: str = "latency",
    ) -> None:
        # Resolve Mongo URI if not explicitly provided.
        if not mongo_uri:
            try:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
                v1 = k8s_client.CoreV1Api()
                svc = v1.read_namespaced_service(
                    name="mongodb-service",
                    namespace="mongodb",
                )
                host = svc.spec.cluster_ip
                mongo_uri = f"mongodb://root:secret@{host}:27017"
            except Exception:
                # Fallback to in-cluster DNS name as a last resort.
                mongo_uri = "mongodb://root:secret@mongodb-service.mongodb:27017"

        self.mongo_uri = mongo_uri
        self.window_slots = max(1, int(window_slots))
        self.latency_db = latency_db

    # ------------------------------------------------------------------
    @staticmethod
    def _coll_to_ip(name: str) -> str:
        return name.replace("_", ".")

    def _iter_latency_docs(self, client: MongoClient) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
        db = client[self.latency_db]
        for coll_name in db.list_collection_names():
            coll = db[coll_name]
            cursor = coll.find({}, {"_id": 0}).sort("slot_anchor", -1).limit(self.window_slots * 4)
            docs = list(cursor)
            if docs:
                yield self._coll_to_ip(coll_name), docs

    @staticmethod
    def _parse_slot(value: Any) -> Optional[int]:
        try:
            return int(str(value).strip())
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _aggregate_node(self, node: str, docs: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
        latest_slot = max(self._parse_slot(doc.get("slot_anchor")) or 0 for doc in docs)
        start_slot = latest_slot - self.window_slots + 1
        # For each anchor we track:
        #  - completed latencies (single_llm_delay)
        #  - number of completed samples
        #  - accumulated lower-bound delay for pending slots (now - start)
        #  - number of pending slots
        #  - skip / timeout / missing counters.
        per_anchor: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "latencies": [],
                "count": 0,
                "skip": 0,
                "r900": 0,
                "missing": 0,
                "pending_lb_total": 0.0,
                "pending_count": 0,
            }
        )

        now_ts = _dt.datetime.now(tz=_dt.timezone.utc).timestamp()

        for doc in docs:
            slot = self._parse_slot(doc.get("slot_anchor"))
            if slot is None or slot < start_slot:
                continue
            # Include all four anchors 1–4 when aggregating latency.
            for anchor_id in ("1", "2", "3", "4"):
                anchor_doc = doc.get(anchor_id, {}) or {}
                bucket = per_anchor[anchor_id]
                if not anchor_doc:
                    bucket["missing"] += 1
                    continue
                if anchor_doc.get("skip"):
                    bucket["skip"] += 1
                    continue
                lat = anchor_doc.get("single_llm_delay")
                start = anchor_doc.get("start")
                if lat is not None:
                    try:
                        lat_val = float(lat)
                    except Exception:
                        bucket["missing"] += 1
                        continue
                    bucket["latencies"].append(lat_val)
                    bucket["count"] += 1
                    if math.isclose(lat_val, 900.0, rel_tol=1e-6) or anchor_doc.get("fail_reason") == "timeout":
                        bucket["r900"] += 1
                else:
                    # No completed delay yet; if we have a start time, treat (now - start)
                    # as a lower bound on the eventual delay and count it as pending.
                    if start is not None:
                        try:
                            start_ts = float(start)
                            elapsed = max(0.0, now_ts - start_ts)
                        except Exception:
                            elapsed = 0.0
                        bucket["pending_lb_total"] += float(elapsed)
                        bucket["pending_count"] += 1
                    else:
                        bucket["missing"] += 1

        return latest_slot, per_anchor

    @staticmethod
    def _compute_stats(records: Dict[str, Any], window_slots: int) -> Dict[str, Any]:
        latencies: List[float] = records.get("latencies", [])
        count = int(records.get("count", 0))
        pending_lb_total = float(records.get("pending_lb_total", 0.0))
        pending_count = int(records.get("pending_count", 0))

        if latencies:
            lat_array = np.array(latencies, dtype=float)
            mean_completed = float(np.mean(lat_array))
            p90_latency = float(np.percentile(lat_array, 90))
            sum_completed = float(np.sum(lat_array))
        else:
            mean_completed = 0.0
            p90_latency = 0.0
            sum_completed = 0.0

        total_obs = count + pending_count
        if total_obs > 0:
            mean_lower_bound = float((sum_completed + pending_lb_total) / float(total_obs))
        else:
            mean_lower_bound = 0.0

        skip = int(records.get("skip", 0))
        r900 = int(records.get("r900", 0))
        missing = int(records.get("missing", 0))
        return {
            # mean_latency is now a conservative lower bound including pending slots.
            "mean_latency": mean_lower_bound,
            "mean_latency_completed": mean_completed,
            "p90_latency": p90_latency,
            "valid_ratio": count / float(window_slots) if window_slots else 0.0,
            "pending_ratio": pending_count / float(window_slots) if window_slots else 0.0,
            "skip_ratio": skip / float(window_slots) if window_slots else 0.0,
            "ratio_900": r900 / float(window_slots) if window_slots else 0.0,
            "missing_ratio": missing / float(window_slots) if window_slots else 0.0,
            "samples": count,
            "pending": pending_count,
        }

    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
            "window_slots": self.window_slots,
        }
        anchors_summary: Dict[str, Dict[str, Any]] = defaultdict(dict)
        nodes: List[str] = []
        latest_seen_slot = 0

        with MongoClient(self.mongo_uri) as client:
            for coll_name, docs in self._iter_latency_docs(client):
                latest_slot, per_anchor = self._aggregate_node(coll_name, docs)
                latest_seen_slot = max(latest_seen_slot, latest_slot)
                if coll_name not in nodes:
                    nodes.append(coll_name)
                for anchor_id, stats in per_anchor.items():
                    anchors_summary[anchor_id][coll_name] = self._compute_stats(stats, self.window_slots)

        meta["latest_slot"] = latest_seen_slot
        anchor_ids = sorted(anchors_summary.keys())

        # Derive global aggregate per anchor
        aggregate: Dict[str, Any] = {}
        for anchor_id in anchor_ids:
            per_node = anchors_summary[anchor_id]
            total_valid = 0
            total_900 = 0
            for node_metrics in per_node.values():
                total_valid += int(node_metrics.get("samples", 0))
                total_900 += node_metrics.get("ratio_900", 0.0) * self.window_slots
            if per_node:
                lat_list = [m["mean_latency"] for m in per_node.values() if m["mean_latency"] > 0]
                mean_latency = float(np.mean(lat_list)) if lat_list else 0.0
            else:
                mean_latency = 0.0
            aggregate[anchor_id] = {
                "mean_latency": mean_latency,
                "valid_ratio": (total_valid / float(len(nodes) * self.window_slots)) if nodes else 0.0,
                "ratio_900": (total_900 / float(len(nodes) * self.window_slots)) if nodes else 0.0,
            }

        snapshot = {
            "meta": meta,
            "nodes": nodes,
            "anchor_ids": anchor_ids,
            "anchors": anchors_summary,
            "aggregate": aggregate,
        }
        return snapshot
