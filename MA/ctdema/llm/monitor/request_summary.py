"""Request distribution summariser for LLM routing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from pymongo import MongoClient

from .snapshot import DEFAULT_MONGO_URI


def _coll_to_ip(name: str) -> str:
    return name.replace("_", ".")


def _slot_filter(start_slot: int, end_slot: int) -> Dict[str, Any]:
    return {
        "$expr": {
            "$and": [
                {
                    "$gte": [
                        {"$toInt": {"$ifNull": ["$metadata.slot_anchor", "0"]}},
                        int(start_slot),
                    ]
                },
                {
                    "$lte": [
                        {"$toInt": {"$ifNull": ["$metadata.slot_anchor", "0"]}},
                        int(end_slot),
                    ]
                },
            ]
        }
    }


def build_request_summary(
    latest_slot: Optional[int],
    window_slots: int,
    mongo_uri: str = DEFAULT_MONGO_URI,
) -> Dict[str, Any]:
    """Aggregate prompt distribution over the recent window."""

    if latest_slot is None or window_slots <= 0:
        return {}

    start_slot = max(0, int(latest_slot) - window_slots + 1)
    per_node: Dict[str, Dict[str, Any]] = {}
    aggregate_anchor_prompts = defaultdict(float)

    with MongoClient(mongo_uri) as client:
        db = client["prompt_recorder"]
        try:
            collections = db.list_collection_names()
        except Exception:
            collections = []
        slot_query = _slot_filter(start_slot, int(latest_slot))

        for coll_name in collections:
            coll = db[coll_name]
            node_ip = _coll_to_ip(coll_name)
            anchor_counts = defaultdict(float)
            total_prompts = 0.0
            total_files = 0
            try:
                cursor = coll.find(slot_query, {"metadata": 1})
            except Exception:
                continue
            for doc in cursor:
                meta = doc.get("metadata", {}) or {}
                anchor = str(meta.get("llm_anchor", "")).strip()
                if not anchor:
                    continue
                size = meta.get("size")
                try:
                    prompts = float(size)
                except (TypeError, ValueError):
                    prompts = 0.0
                anchor_counts[anchor] += prompts
                aggregate_anchor_prompts[anchor] += prompts
                total_prompts += prompts
                total_files += 1

            if total_files == 0:
                continue

            per_node[node_ip] = {
                "total_prompts": total_prompts,
                "total_files": total_files,
                "anchors": {},
            }
            for anchor, count in anchor_counts.items():
                share = count / total_prompts if total_prompts > 0 else 0.0
                per_node[node_ip]["anchors"][anchor] = {
                    "prompts": count,
                    "share": share,
                }

    aggregate = {}
    total_all_prompts = sum(aggregate_anchor_prompts.values())
    for anchor, count in aggregate_anchor_prompts.items():
        aggregate[anchor] = {
            "prompts": count,
            "share": count / total_all_prompts if total_all_prompts > 0 else 0.0,
        }

    if not per_node and not aggregate:
        return {}

    return {
        "per_node": per_node,
        "aggregate": aggregate,
        "window_slots": window_slots,
        "slot_range": [start_slot, int(latest_slot)],
    }
