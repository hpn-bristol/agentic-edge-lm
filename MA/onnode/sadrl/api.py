import ast
# Monkey-patch kubernetes.config.load_incluster_config before anything else
# imports it. See sadrl/k8s_compat.py for why the lib's built-in is broken
# on this RKE cluster.
from sadrl.k8s_compat import patch_load_incluster_config as _patch_k8s
_patch_k8s()
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import uvicorn
import netifaces as ni
import shutil
import os
from PIL import Image
import json
import tarfile
import re
import time
from typing import List, Dict, Any, Optional
from sadrl.SevEnv_no_GPU import *
from sadrl.SevEnv_single_GPU import *
from sadrl.SevEnv_double_GPUs import *
from kubernetes import client, config
import threading
import queue
from sadrl.gpt_local_agent import GPTLocalAgent  # type: ignore
try:
    from sadrl.Lyapunov_agent import LyapunovAgent
except Exception:
    LyapunovAgent = None
try:
    from sadrl.agent_event_logger import AgentEventLogger
except Exception:
    try:
        from agent_event_logger import AgentEventLogger  # type: ignore
    except Exception:
        AgentEventLogger = None  # type: ignore

# Cross-layer mismatch telemetry: module-level logger + state.
# Updated each /trigger after the local agent (Lyapunov, GPT, etc.) chooses
# an action; consulted in /uploadfile to mark whether arriving prompts'
# target LM is currently deployed on this node.
_runtime_event_logger = AgentEventLogger("local") if AgentEventLogger is not None else None
_active_lms_now: List[int] = []          # e.g. [1, 3, 4] (1-indexed LM ids)
_current_agent_mode: str = "unknown"     # 'lyapunov' | 'gpt'
import logging
from logging.handlers import RotatingFileHandler
import subprocess
from pymongo.errors import DuplicateKeyError

import asyncio
import aiofiles
import httpx
import pandas as pd
import numpy as np

from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorCollection 

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


handler = RotatingFileHandler(
    "app.log", maxBytes=10*1024*1024, backupCount=5
)

console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.DEBUG) 

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[handler, console_handler]
) # kubectl logs -f <pod-name> -n allworkers can be used to monitor the status of all logs real time

logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)

logging.getLogger("pymongo").setLevel(logging.WARNING) # remove the stuff that is useless for debug
logging.getLogger("kubernetes.client.rest").setLevel(logging.WARNING)  # remove the stuff that is useless for debug
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI() # The range of valid ports of node is 30000-32767, we will use 30000

dbname = "uninitialized"
db_ready = False
mongo = None
observe = 0
explore = 0
memory_length = 0
batch_size = 0
length_of_slot = 0

server = None
agent = None
training_context = None
local_ip_address = None
prompt_col = None
latency_col = None
local_gpt_scheduler_task = None  # background task for GPTLocalAgent, if enabled
node_layout_intent_local: Dict[str, Any] = {}  # macro role for this node in current epoch

# See sadrl/k8s_compat.py for why we bypass config.load_incluster_config()
# on this RKE cluster.
from sadrl.k8s_compat import get_api_client as _get_api_client, get_mongo_ip as _get_mongo_ip
mongo = AsyncIOMotorClient(f"mongodb://root:secret@{_get_mongo_ip()}:27017")
k8s_client = _get_api_client()
v1 = client.CoreV1Api(api_client=k8s_client)

generated_files = {}

pod_name = os.getenv('HOSTNAME')
namespace = os.getenv("POD_NAMESPACE")
node_name = os.getenv("NODE_NAME")
if node_name:
    node = v1.read_node(node_name)
else:
    node = None

ANCHOR_LIST = ["1","2","3","4"]
# ----------------------------------------------------

def _select_agent_class(mode: str):
    agent_mode = (mode or "lyapunov").lower()
    if agent_mode == "lyapunov":
        if LyapunovAgent is None:
            raise RuntimeError("LyapunovAgent is unavailable in this image")
        return LyapunovAgent, "lyapunov"
    if agent_mode in ("gpt", "gpt_local", "llm"):
        return GPTLocalAgent, "gpt"
    raise ValueError(
        f"Unsupported SADRL_AGENT mode '{mode}'. "
        "Supported modes are: lyapunov, gpt."
    )

def initialize_system(dbname, observe, explore, memory_length, batch_size, length_of_slot):
    global server, agent, training_context, local_ip_address, prompt_col, latency_col, node_name  

    # Robust GPU detection: treat "0" (string) or 0 as no-GPU.
    gpu_raw = node.status.allocatable.get('nvidia.com/gpu') if node else None
    try:
        gpu_count = int(str(gpu_raw)) if gpu_raw is not None else 0
    except Exception:
        gpu_count = 0

    if gpu_count > 0:
        if node_name in {"worker5", "worker8"}:
            server = SevEnv_double_GPUs(dbname, length_of_slot, cooldown_period=0)
            state_dims = server.llm_amount * 2 + 3
            n_actions = len(server.action_combo)
        else:
            server = SevEnv_single_GPU(dbname, length_of_slot, cooldown_period=0)
            state_dims = server.llm_amount * 2 + 3
            n_actions = len(server.action_combo)
    else:
        server = SevEnv_no_GPU(dbname, length_of_slot, cooldown_period=0)
        state_dims = server.llm_amount * 2 + 2
        n_actions = len(server.action_combo)
   
    local_ip_address = server.node_ip

    prompt_col = mongo['prompt_recorder'][f"{local_ip_address.replace('.', '_')}"]
    latency_col = mongo['latency'][f"{local_ip_address.replace('.', '_')}"]
    
    chkpt_dir = os.path.join('/app/models')
    load_dir = os.path.join('/app/models')
    lr = 0.0002
    gamma = 0.99
    epsilon = 0.9

    # Default to Lyapunov agent unless explicitly overridden.
    agent_choice = os.getenv("SADRL_AGENT", "lyapunov").lower()
    anchors = ["1","2"] if getattr(server, "llm_amount", 4) == 2 else ["1","2","3","4"]
    AgentCls, normalized_mode = _select_agent_class(agent_choice)

    agent_kwargs: Dict[str, Any] = dict(
        state_dim=state_dims,
        action_dim=n_actions,
        chkpt_dir=chkpt_dir,
        load_dir=load_dir,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        batch_size=batch_size,
        capacity=memory_length,
        action_combo=getattr(server, "action_combo", None),
        anchors=anchors,
    )
    # If we are using the GPT local agent, pass the current macro role (if any).
    if AgentCls is GPTLocalAgent:
        agent_kwargs["node_role"] = node_layout_intent_local

    agent_obj = AgentCls(**agent_kwargs)
    training_context_obj = TrainingContext(server, agent_obj, observe, explore)

    agent = agent_obj
    training_context = training_context_obj
    globals()["_current_agent_mode"] = normalized_mode

    # Forward the per-run dbname to the local agent's event logger so its
    # success / parse_fail / fallback events land in agent_events.<dbname>
    # alongside the rest of this run's collections.
    if AgentCls is GPTLocalAgent:
        try:
            agent_obj.set_dbname(dbname)
        except Exception:
            logger.warning("[AGENT] set_dbname on GPTLocalAgent failed", exc_info=True)

    logger.info(
        "[AGENT] System initialized on node_ip=%s with SADRL_AGENT='%s' (class=%s)",
        local_ip_address,
        agent_choice,
        AgentCls.__name__,
    )
    logger.info("System initialized with the new parameters.")


async def _run_gpt_local_scheduler(period_sec: float = 30.0):
    """
    Background loop for GPTLocalAgent: drive local decisions on its own clock.

    Once started, this loop periodically calls TrainingContext.step with an
    increasing local_slot_anchor, independent of the external /trigger cadence.
    """
    global training_context, db_ready
    local_slot_anchor = 0
    logger.info(f"[GPT-LOCAL] starting background scheduler loop with period={period_sec}s")
    while db_ready and training_context is not None:
        try:
            await training_context.step(local_slot_anchor, status="routing", incre_from=0)
        except Exception as e:
            logger.error(f"[GPT-LOCAL] Error during scheduler step (local_slot_anchor={local_slot_anchor}): {e}", exc_info=True)
        local_slot_anchor += 1
        try:
            await asyncio.sleep(period_sec)
        except Exception:
            break
    logger.info("[GPT-LOCAL] scheduler loop stopped.")
    
class DbnamePayload(BaseModel):
    dbname: str
    observe: int
    explore: int
    memory_length: int
    batch_size: int
    length_of_slot: int
@app.post("/get_dbname")
async def get_dbname_endpoint(payload: DbnamePayload):
    global dbname, db_ready, mongo, observe, explore, memory_length, batch_size, length_of_slot
    dbname = payload.dbname
    observe = payload.observe
    explore = payload.explore
    memory_length = payload.memory_length
    batch_size = payload.batch_size
    length_of_slot = payload.length_of_slot
    try:
        initialize_system(dbname, observe, explore, memory_length, batch_size, length_of_slot)
        db_ready = True
        # Align telemetry with the current run database name.
        if _runtime_event_logger is not None:
            try:
                _runtime_event_logger.set_dbname(dbname)
            except Exception:
                logger.warning("[telemetry] runtime event logger setup failed", exc_info=True)
        return {
            "status": "success",
            "dbname": dbname,
            "observe": observe,
            "explore": explore,
            "memory_length": memory_length,
            "batch_size": batch_size,
            "length_of_slot": length_of_slot
        }
    except Exception as e:
        logger.error("DB initialization failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB initialization failed: {str(e)}")

@app.post("/check_db_ready")
async def check_db_ready():
    global db_ready
    if db_ready:
        return {"status": "ready"}
    else:
        return {"status": "not_ready"}

# ---- runtime agent mode switch ----
class AgentModePayload(BaseModel):
    # 'lyapunov' | 'gpt'
    mode: str

@app.post("/set_agent_mode")
async def set_agent_mode(p: AgentModePayload):
    global agent, training_context, local_gpt_scheduler_task, _current_agent_mode
    if not db_ready or server is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    mode = (p.mode or "").lower()
    try:
        # infer dims from current server
        has_gpu = getattr(server, 'vGPU', 0) and getattr(server, 'vGPU', 0) > 0
        state_dims = server.llm_amount * 2 + (3 if has_gpu else 2)
        n_actions = len(getattr(server, 'action_combo', []))
        chkpt_dir = os.path.join('/app/models')
        load_dir = os.path.join('/app/models')
        lr = 0.0002
        gamma = 0.99
        epsilon = 0.9

        anchors = ["1","2"] if getattr(server, "llm_amount", 4) == 2 else ["1","2","3","4"]
        AgentCls, normalized_mode = _select_agent_class(mode)

        agent_kwargs: Dict[str, Any] = dict(
            state_dim=state_dims,
            action_dim=n_actions,
            chkpt_dir=chkpt_dir,
            load_dir=load_dir,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            batch_size=batch_size,
            capacity=memory_length,
            action_combo=getattr(server, "action_combo", None),
            anchors=anchors,
        )
        if AgentCls is GPTLocalAgent:
            agent_kwargs["node_role"] = node_layout_intent_local

        new_agent = AgentCls(**agent_kwargs)
        training_context.agent = new_agent
        agent = new_agent
        _current_agent_mode = normalized_mode
        if AgentCls is GPTLocalAgent:
            try:
                new_agent.set_dbname(dbname)
            except Exception:
                logger.warning("[AGENT] set_dbname on swapped GPTLocalAgent failed", exc_info=True)
        logger.info(
            "[AGENT] Switched agent mode to '%s' (class=%s, node_ip=%s)",
            normalized_mode,
            AgentCls.__name__,
            local_ip_address,
        )
        return {"status": "success", "mode": normalized_mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to switch agent mode", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to switch agent: {e}")

# ------------------------------ Lyapunov params setter ---------------------------------------
class HeurParams(BaseModel):
    lambda_churn: float | None = None
    kappa_workload: float | None = None
    price_p0: float | None = None
    price_p1: float | None = None
    price_p2: float | None = None
    text_gpu_speedup: float | None = None
    coverage_bonus: float | None = None
    coverage_threshold: float | None = None
    priority: dict[str, float] | None = None
    # Per-LM modality weights (4-list, anchors 1..4). Image LMs entries on
    # alpha_cpu must remain 0.0; the calibration driver enforces this. These
    # fields let the SPSA driver freeze α on every pod after profiling
    # without requiring a pod restart.
    alpha_cpu: list[float] | None = None
    alpha_gpu: list[float] | None = None

@app.post("/set_lyapunov_params")
async def set_lyapunov_params(p: HeurParams):
    if not db_ready or agent is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    if LyapunovAgent is None or not isinstance(agent, LyapunovAgent):
        return {"status": "ignored", "reason": "agent is not LyapunovAgent"}

    updated = {}

    def _assign(attr: str, value: float | dict[str, float]):
        setattr(agent, attr, value)
        updated[attr] = value

    if p.lambda_churn is not None:
        _assign("lambda_churn", float(p.lambda_churn))
    if p.kappa_workload is not None:
        _assign("kappa_workload", float(p.kappa_workload))
    if p.price_p0 is not None:
        _assign("price_p0", float(p.price_p0))
    if p.price_p1 is not None:
        _assign("price_p1", float(p.price_p1))
    if p.price_p2 is not None:
        _assign("price_p2", float(p.price_p2))
    if p.text_gpu_speedup is not None and hasattr(agent, "text_gpu_speedup"):
        _assign("text_gpu_speedup", float(p.text_gpu_speedup))

    if LyapunovAgent is not None and isinstance(agent, LyapunovAgent):
        if p.coverage_bonus is not None:
            _assign("coverage_bonus", float(p.coverage_bonus))
        if p.coverage_threshold is not None:
            _assign("coverage_threshold", float(p.coverage_threshold))
        if p.priority is not None:
            try:
                prio = {str(k): float(v) for k, v in p.priority.items()}
            except Exception:
                prio = {}
            _assign("priority", prio)

    # Per-LM modality weights used by the Lyapunov service-rate proxy.
    if p.alpha_cpu is not None and len(p.alpha_cpu) >= 4:
        try:
            vec = [float(v) for v in p.alpha_cpu[:4]]
        except Exception:
            vec = None
        if vec is not None:
            if hasattr(agent, "alpha_cpu"):
                _assign("alpha_cpu", vec)
    if p.alpha_gpu is not None and len(p.alpha_gpu) >= 4:
        try:
            vec = [float(v) for v in p.alpha_gpu[:4]]
        except Exception:
            vec = None
        if vec is not None:
            if hasattr(agent, "alpha_gpu"):
                _assign("alpha_gpu", vec)

    logger.info(f"[LYAPUNOV-PARAMS] updated ({agent.__class__.__name__}): {updated}")
    return {"status": "success", "updated": updated}


@app.get("/get_local_state")
async def get_local_state():
    """Expose the local deployment agent's most recent state vector.

    Returns the *post-action* state (server.step()'s next_state) so that the
    backlog Q_i reflects the cluster after the agent's deployment change for
    the current slot has been applied. Falls back to pre-action state if no
    full step has completed yet (early calibration warmup).

    State layout follows SevEnv:
        [pod_status (4), backlog (4), resources (3)]   for 4-anchor envs,
        [pod_status (2), backlog (2), resources (2)]   for 2-anchor envs.
    """
    if not db_ready or training_context is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    # Prefer post-action snapshot; fall back to pre-action only if the very
    # first step hasn't completed yet.
    state = training_context.last_next_state
    state_source = "post_action"
    if state is None:
        state = training_context.state
        state_source = "pre_action"
    if state is None:
        return {
            "status": "warmup",
            "state_source": None,
            "slot_anchor": None,
            "state": None,
            "backlog": None,
            "ts": None,
        }
    state_list = [float(v) for v in state]
    n = len(state_list)
    if n >= 11:
        backlog = state_list[4:8]
    elif n >= 6:
        backlog = state_list[2:4]
    else:
        backlog = []
    return {
        "status": "ok",
        "state_source": state_source,
        "slot_anchor": int(training_context.last_slot_anchor) if training_context.last_slot_anchor is not None else None,
        "state": state_list,
        "backlog": backlog,
        "ts": float(training_context.last_state_ts) if training_context.last_state_ts is not None else None,
    }


# -------------------------------------
# Macro role / layout intent from global planner (per node)
# -------------------------------------

class NodeLayoutIntentPayload(BaseModel):
    layout: Dict[str, Any]


@app.post("/set_node_layout_intent")
async def set_node_layout_intent(p: NodeLayoutIntentPayload):
    """Update this node's macro role (layout intent) for the current epoch."""
    global node_layout_intent_local, agent
    node_layout_intent_local = p.layout or {}
    if isinstance(agent, GPTLocalAgent):
        try:
            agent.update_node_role(node_layout_intent_local)
        except Exception as e:
            logger.error(f"[NODE-ROLE] Failed to update GPTLocalAgent node_role: {e}", exc_info=True)
            return {"status": "error", "detail": str(e)}
    logger.info(f"[NODE-ROLE] updated node_layout_intent_local={node_layout_intent_local}")
    return {"status": "success"}

# -------------------------------------
# generate the prompts; initialize the latency; update local json file; send 'success' back to sender once received
# -------------------------------------
async def async_write_lines(file_path, lines):
    async with aiofiles.open(file_path, 'a') as f:
        for line in lines:
            await f.write(line + '\n')

async def async_write_tar(image_text_pth, image, count):
    def sync_write_tar():
        # mode = 'w'
        with tarfile.open(image_text_pth, 'w') as tar:
            for i in range(count):
                tar.add(image, arcname=f"image_{i + 1}")
    await asyncio.to_thread(sync_write_tar)

async def ensure_files_ready(file_paths, llm_sizes, max_retries=10, delay=0.1):
    for attempt in range(max_retries):
        all_ok = True
        for idx, path in enumerate(file_paths):
            size = llm_sizes[idx]
            if size == 0:
                # zero-file is allowed as long as the file physically exists
                if not os.path.exists(path):
                    all_ok = False
                    break
            else:
                # Must exist and not be empty
                if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
                    all_ok = False
                    break
        if all_ok:
            return
        await asyncio.sleep(delay)

    raise Exception("Some files are not ready after retries.")


async def generate_files(llm1, llm2, llm3, llm4, slot_anchor, prompt_col: AsyncIOMotorCollection, latency_col: AsyncIOMotorCollection, local_ip_address):
    logger = logging.getLogger(__name__)
    logger.debug("Generating files for llm prompts.")
    local_ip_address_underscore = local_ip_address.replace(".", "_")
    t_t_sentence = 'can you explain multi-access technology in wireless network'

    file_paths = []
    file_sizes = []

    # anchor 1
    if "1" in ANCHOR_LIST: 
        file_name_1 = f"{slot_anchor}_{local_ip_address_underscore}_1_{llm1}.txt"
        t_t_s_prompts_pth = "/app/prompt_queue/" + file_name_1
        await async_write_lines(t_t_s_prompts_pth, [t_t_sentence]*llm1)
        file_paths.append(t_t_s_prompts_pth)
        file_sizes.append(llm1)
    else:
        file_name_1 = None

    # anchor 2: still create an empty file so that per-slot bookkeeping
    # (file counts, start timestamps) remains consistent, but llm2 workload
    # is effectively zero because llm2==0.
    if "2" in ANCHOR_LIST:
        file_name_2 = f"{slot_anchor}_{local_ip_address_underscore}_2_{llm2}.txt"
        t_t_b_prompts_pth = "/app/prompt_queue/" + file_name_2
        await async_write_lines(t_t_b_prompts_pth, [t_t_sentence]*llm2)
        file_paths.append(t_t_b_prompts_pth)
        file_sizes.append(llm2)
    else:
        file_name_2 = None

    # anchor 3
    if "3" in ANCHOR_LIST: 
        file_name_3 = f"{slot_anchor}_{local_ip_address_underscore}_3_{llm3}.tar"
        image = "/app/sadrl/text_image.jpg"
        image_text_pth = "/app/prompt_queue/" + file_name_3
        await async_write_tar(image_text_pth, image, llm3)
        file_paths.append(image_text_pth)
        file_sizes.append(llm3)
    else:
        file_name_3 = None

    # anchor 4
    if "4" in ANCHOR_LIST: 
        t_i_sentence = 'draw a man hold a basketball'
        file_name_4 = f"{slot_anchor}_{local_ip_address_underscore}_4_{llm4}.txt"
        text_image_pth = "/app/prompt_queue/" + file_name_4
        await async_write_lines(text_image_pth, [t_i_sentence]*llm4)
        file_paths.append(text_image_pth)
        file_sizes.append(llm4)
    else:
        file_name_4 = None

    await ensure_files_ready(file_paths, file_sizes)

    metadata_list = []
    file_name_list = []
    if "1" in ANCHOR_LIST and file_name_1:
        metadata_list.append({
            'slot_anchor': str(slot_anchor),
            'ip': local_ip_address,
            'llm_anchor': '1',
            'size': str(llm1)
        })
        file_name_list.append(file_name_1)
    if "2" in ANCHOR_LIST and file_name_2:
        metadata_list.append({
            'slot_anchor': str(slot_anchor),
            'ip': local_ip_address,
            'llm_anchor': '2',
            'size': str(llm2)
        })
        file_name_list.append(file_name_2)
    if "3" in ANCHOR_LIST and file_name_3:
        metadata_list.append({
            'slot_anchor': str(slot_anchor),
            'ip': local_ip_address,
            'llm_anchor': '3',
            'size': str(llm3)
        })
        file_name_list.append(file_name_3)
    if "4" in ANCHOR_LIST and file_name_4:
        metadata_list.append({
            'slot_anchor': str(slot_anchor),
            'ip': local_ip_address,
            'llm_anchor': '4',
            'size': str(llm4)
        })
        file_name_list.append(file_name_4)

    for file_name, metadata in zip(file_name_list, metadata_list):
        await prompt_col.update_one(
            {"file_name": file_name}, 
            {
                "$set": {"metadata": metadata, "file_name": file_name},
            }, 
            upsert=True
        )

    current_time = time.time()
    updates = {}
    for anchor_id, size_val in zip(["1","2","3","4"], [llm1, llm2, llm3, llm4]):
        if anchor_id not in ANCHOR_LIST: 
            continue
        if size_val == 0:
            updates[f"{anchor_id}.skip"] = True
        updates[f"{anchor_id}.start"] = current_time

    updates["slot_anchor"] = str(slot_anchor)

    await latency_col.update_one(
        {"slot_anchor": str(slot_anchor)},
        {"$set": updates},
        upsert=True
    )

    slot_str = str(slot_anchor)
    interval = 1.0
    while True:
        doc = await latency_col.find_one({"slot_anchor": slot_str})
        if doc is not None:
            # Wait until start timestamps have been written for all anchors.
            starts_ok = True
            for anchor_id in ANCHOR_LIST:
                if anchor_id not in doc or "start" not in doc[anchor_id]:
                    starts_ok = False
                    break
            if starts_ok:
                break
        await asyncio.sleep(interval)

    logger.info(f"Files generated successfully for slot_anchor={slot_anchor}.")

class CheckFilesPayload(BaseModel):
    slot_anchor: int

@app.post("/check_files_ready")
async def check_files_ready_endpoint(payload: CheckFilesPayload):
    slot_anchor = payload.slot_anchor
    slot_prefix = f"{slot_anchor}_"
    files_for_slot = [fname for fname in os.listdir(UPLOAD_DIR) if fname.startswith(slot_prefix)]
    file_count = len(files_for_slot)

    if file_count < len(ANCHOR_LIST): 
        return {"status": "not_ready", "count": file_count}

    projection = {"_id": 0}
    for anchor_id in ANCHOR_LIST:
        projection[f"{anchor_id}.start"] = 1

    query = {"slot_anchor": str(slot_anchor)}
    result = await latency_col.find_one(query, projection)
    if result is None:
        return {"status": "not_ready", "count": file_count}

    have_all_starts = True
    for anchor_id in ANCHOR_LIST:
        start_val = result.get(anchor_id, {}).get("start")
        if start_val is None:
            have_all_starts = False
            break
    if not have_all_starts:
        return {"status": "not_ready", "count": file_count}

    db_count = 0
    for fname in files_for_slot:
        doc = await prompt_col.find_one({"file_name": fname})
        if doc:
            db_count += 1

    if db_count < len(ANCHOR_LIST): 
        return {"status": "not_ready", "count": file_count}

    generated_files[slot_anchor] = {"ready": True}
    return {"status": "ready", "count": file_count}

class NumberPayload(BaseModel):
    llm1: int
    llm2: int
    llm3: int
    llm4: int
    slot_anchor: int

@app.post("/generate")
async def generate(payload: NumberPayload):
    logger.info(f"【received requests】slot={payload.slot_anchor}, ip={local_ip_address}, "
                f"llm=({payload.llm1},{payload.llm2},{payload.llm3},{payload.llm4})")
    if not db_ready or prompt_col is None or latency_col is None or local_ip_address is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")

    unique_id = f"{payload.slot_anchor}_{local_ip_address}"

    doc_to_insert = {
        "_id": unique_id,
        "metadata": {
            "slot_anchor": str(payload.slot_anchor),
            "ip": local_ip_address
        },
        "file_names": [],
        "insert_time": time.time()
    }

    try:
        insert_result = await prompt_col.insert_one(doc_to_insert)
        logger.info(f"[Generate] slot={payload.slot_anchor}, ip={local_ip_address}, "
                    f"inserted_id={insert_result.inserted_id}, allow generation.")
    except DuplicateKeyError:
        logger.info(f"[Generate] slot={payload.slot_anchor}, ip={local_ip_address}, "
                    f"skip - already generated or in progress.")
        query = {"slot_anchor": str(payload.slot_anchor)}
        doc_in_latency = await latency_col.find_one(query)
        if not doc_in_latency or "1" not in doc_in_latency:
            current_time = time.time()
            updates = {}
            for anchor_id in ANCHOR_LIST:
                updates[f"{anchor_id}.start"] = current_time
            updates["slot_anchor"] = str(payload.slot_anchor)

            await latency_col.update_one(
                {"slot_anchor": str(payload.slot_anchor)},
                {"$set": updates},
                upsert=True
            )
        return {"status": "success", "message": "Already generated or in progress."}
    except Exception as e:
        logger.error(f"Error when inserting doc: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        logger.debug(
            f"Start generating files for slot={payload.slot_anchor}, "
            f"llm1={payload.llm1}, llm2={payload.llm2}, llm3={payload.llm3}, llm4={payload.llm4}"
        )
        await generate_files(
            payload.llm1, payload.llm2, payload.llm3, payload.llm4,
            payload.slot_anchor, prompt_col, latency_col, local_ip_address
        )
        return {"status": "success", "message": "Files generated successfully."}
    except Exception as e:
        logger.error(f"Error generating files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

TARGET_DIRECTORY = "/app/"
DIRECTORIES_TO_CLEAN = ["current_prompt", "output_local", "output_relay", "prompt_queue", "models", "results"]

@app.post("/cleanup")
async def cleanup():
    if not db_ready:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    try:
        if not os.path.exists(TARGET_DIRECTORY):
            logger.error(f"Directory {TARGET_DIRECTORY} does not exist.")
            raise HTTPException(status_code=404, detail=f"Directory {TARGET_DIRECTORY} does not exist.")

        for dir_name in DIRECTORIES_TO_CLEAN:
            dir_path = os.path.join(TARGET_DIRECTORY, dir_name)
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                logger.warning(f"Directory {dir_path} does not exist or is not a directory. Skipping.")
                continue

            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                        logger.info(f"Deleted file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        logger.info(f"Deleted directory: {item_path}")
                except Exception as file_error:
                    logger.warning(f"Failed to delete using Python methods, trying system command: {file_error}")
                    try:
                        subprocess.run(["rm", "-rf", item_path], check=True)
                        logger.info(f"Forcefully deleted: {item_path}")
                    except subprocess.CalledProcessError as cmd_error:
                        logger.error(f"Failed to forcefully delete {item_path}: {cmd_error}")

        logger.info("Cleanup completed. Specified directories have been cleared.")
        return {"status": "success", "message": "Specified directories have been cleared."}

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during cleanup. Check logs for details.")


DIRECTORIES_TO_CLEAN_PROMPT = ["current_prompt", "output_local", "output_relay", "prompt_queue"]
@app.post("/cleanup_prompt")
async def cleanup_prompt():
    if not db_ready:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    try:
        if not os.path.exists(TARGET_DIRECTORY):
            logger.error(f"Directory {TARGET_DIRECTORY} does not exist.")
            raise HTTPException(status_code=404, detail=f"Directory {TARGET_DIRECTORY} does not exist.")

        for dir_name in DIRECTORIES_TO_CLEAN_PROMPT:
            dir_path = os.path.join(TARGET_DIRECTORY, dir_name)
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                logger.warning(f"Directory {dir_path} does not exist or is not a directory. Skipping.")
                continue

            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                        logger.info(f"Deleted file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        logger.info(f"Deleted directory: {item_path}")
                except Exception as file_error:
                    logger.warning(f"Failed to delete using Python methods, trying system command: {file_error}")
                    try:
                        subprocess.run(["rm", "-rf", item_path], check=True)
                        logger.info(f"Forcefully deleted: {item_path}")
                    except subprocess.CalledProcessError as cmd_error:
                        logger.error(f"Failed to forcefully delete {item_path}: {cmd_error}")

        logger.info("Cleanup prompt file only, completed. Specified directories have been cleared.")
        return {"status": "success", "message": "Specified directories have been cleared."}

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during cleanup. Check logs for details.")
    

class CleanupSlotPayload(BaseModel):
    slot_anchor: int

@app.post("/cleanup_slot")
async def cleanup_slot(payload: CleanupSlotPayload):
    slot_anchor = payload.slot_anchor
    prefix = f"{slot_anchor}_"

    await prompt_col.delete_many({"metadata.slot_anchor": str(slot_anchor)})
    await latency_col.delete_one({"slot_anchor": str(slot_anchor)})

    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(prefix):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)

    while True:
        still_exists = any(
            fn.startswith(prefix) for fn in os.listdir(UPLOAD_DIR)
        )
        if not still_exists:
            break
        logger.info(f"Still see some files of slot={slot_anchor}, wait a bit more...")
        await asyncio.sleep(0.5)

    if slot_anchor in generated_files:
        del generated_files[slot_anchor]

    return {"status": "success", "message": f"Slot {slot_anchor} cleaned."}

UPLOAD_DIR = "/app/prompt_queue"

@app.post("/uploadfile")
async def upload_file(files: UploadFile = File(...), data: str = Form(...)):
    logger.debug(f"uploadfile called: db_ready={db_ready}, prompt_col={prompt_col}, latency_col={latency_col}, local_ip_address={local_ip_address}, dbname={dbname}")

    if not db_ready or prompt_col is None or latency_col is None or local_ip_address is None:
        logger.error("System not initialized yet before handling uploadfile request.")
        raise HTTPException(status_code=400, detail="System not initialized yet.")

    try:
        filename = files.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        logger.debug(f"Uploading file: {filename} to {file_path}")

        async with aiofiles.open(file_path, "wb") as f:
            file_bytes = await files.read() 
            await f.write(file_bytes)

        pattern = r"^(?P<slot_anchor>\d+)_(?P<sender_ip_underscore>(?:\d+_){3}\d+)_(?P<llm_anchor>\w+)_(?P<size>\d+)(\.txt|\.tar)$"
        match = re.match(pattern, filename)
        if match:
            metadata = {
                'slot_anchor': match.group('slot_anchor'),
                'ip': match.group('sender_ip_underscore').replace("_", "."),
                'llm_anchor': match.group('llm_anchor'),
                'size': match.group('size')
            }
        else:
            logger.warning(f"Filename '{filename}' does not match the expected pattern.")
            raise ValueError(f"Filename '{filename}' does not match expected pattern.")

        # If this node does not support the received anchor (e.g., image anchors 3/4 on CPU-only),
        # reject gracefully: delete the file locally and return success to avoid blocking sender flow.
        if metadata['llm_anchor'] not in ANCHOR_LIST:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Rejected unsupported anchor {metadata['llm_anchor']} on this node. Deleted file: {file_path}")
            except Exception as del_err:
                logger.warning(f"Failed to delete unsupported anchor file {file_path}: {del_err}")

            receive_time = time.time()
            llm_anchor = metadata['llm_anchor']
            slot_anchor = metadata['slot_anchor']
            # Do NOT upsert into prompt_col; just acknowledge receipt for uplink accounting on sender side.
            return {"status": "success", 'receive_time': receive_time, 'llm_anchor': llm_anchor, 'slot_anchor': slot_anchor}

        await prompt_col.update_one(
            {"file_name": filename},
            {
                "$set": {"metadata": metadata, "assigned": True, "file_name": filename},
            },
            upsert=True
        )

        logger.debug(f"Database updated for file: {files.filename} at {time.time()}")

        receive_time = time.time()

        # A cross-node prompt has arrived for llm_anchor on this node.
        # Compare it with the LM set currently active here, snapshotted at
        # the most recent /trigger. If the target LM is not active, routing
        # and deployment are temporarily misaligned.
        try:
            target_lm_int = int(metadata['llm_anchor'])
            was_active = target_lm_int in _active_lms_now
            if _runtime_event_logger is not None:
                _runtime_event_logger.log(
                    outcome="success",
                    slot_index=int(metadata['slot_anchor']),
                    node=node_name or "",
                    kind="prompt_received",
                    target_lm=target_lm_int,
                    active_lms_now=list(_active_lms_now),
                    was_active=bool(was_active),
                    sender_ip=metadata.get('ip'),
                    agent_mode=_current_agent_mode,
                )
        except Exception:
            # Telemetry must never break /uploadfile; swallow.
            pass

        packaged_data = json.loads(data)
        llm_anchor = packaged_data.get('llm_anchor')
        slot_anchor = match.group('slot_anchor')

        logger.info(f"File {filename} from {metadata['ip']} received at time {receive_time}.")

        return {"status": "success", 'receive_time': receive_time, 'llm_anchor': llm_anchor, 'slot_anchor': slot_anchor}
    
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def async_find_file(base_dir, slot_anchor, sender_ip, llm_anchor):
    def sync_find_file():
        file_path = f"{slot_anchor}_{sender_ip}_{llm_anchor}"
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.startswith(file_path):
                    full_path = os.path.join(root, filename)
                    return full_path, filename
        return None
    return await asyncio.to_thread(sync_find_file)

async def post_with_retries(url, json=None, max_retries=10, retry_delay=0.5, **kwargs):
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=json, timeout=60, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} to {url} failed with HTTP status {e.response.status_code}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Returning None.")
                    return None
            except httpx.RequestError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} to {url} failed with request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Returning None.")
                    return None

async def post_files_with_retries(url, files, data=None, max_retries=10, retry_delay=1, **kwargs):
    async with httpx.AsyncClient(verify=False) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, files=files, data=data, timeout=60, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} to {url} failed with HTTP status {e.response.status_code}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Returning None.")
                    return None
            except httpx.RequestError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} to {url} failed with request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Returning None.")
                    return None

def force_delete_file(file_path):
    try:
        os.remove(file_path)
        logger.info(f"Deleted file: {file_path}")
    except OSError as os_error:
        logger.warning(f"Failed to delete file {file_path} using os.remove: {os_error}. Trying system command.")
        try:
            subprocess.run(["rm", "-rf", file_path], check=True)
            logger.info(f"Forcefully deleted file: {file_path}")
        except subprocess.CalledProcessError as cmd_error:
            logger.error(f"Failed to forcefully delete file {file_path}: {cmd_error}")

async def async_force_delete_file(file_path):
    await asyncio.to_thread(force_delete_file, file_path)

async def async_delete_with_retries(collection, query, max_retries=5, retry_delay=0.5):
    from pymongo.errors import PyMongoError

    for attempt in range(max_retries):
        try:
            result = await collection.delete_one(query)
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted MongoDB record: {query}")
                return True
            else:
                logger.warning(f"No document matched for deletion: {query}")
                return False
        except PyMongoError as e:
            logger.error(f"Error deleting document on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying deletion after {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Exceeded maximum retries for deleting document: {query}")
                return False

async def send_prompts_to_others(destination, slot_anchor, prompt_col, latency_col, sender_ip):
    logger.debug(f"[send_prompts_to_others] Start. slot_anchor={slot_anchor}")
    base_dir = '/app/prompt_queue'
    responses = []
    processed_files = set()

    for i, receiver_url in enumerate(destination):
        if i >= len(ANCHOR_LIST):
            break

        llm_anchor = ANCHOR_LIST[i]
        sender_ip_underscore = sender_ip.replace(".", "_")

        logger.debug(
            f"[send_prompts_to_others] i={i}, llm_anchor={llm_anchor}, receiver_url={receiver_url}"
        )
        
        if receiver_url is None:
            logger.debug(f"[send_prompts_to_others] destination[{i}] is None => skip.")
            continue

        result = await async_find_file(base_dir, slot_anchor, sender_ip_underscore, llm_anchor)
        logger.debug(f"[send_prompts_to_others] Searching for file matching slot={slot_anchor}, ip={sender_ip_underscore}, anchor={llm_anchor} in {base_dir}")

        if result is None:
            logger.warning(f"[send_prompts_to_others] No file found for slot={slot_anchor}, llm_anchor={llm_anchor}")
            continue

        file_path, file_name = result
        if file_name in processed_files:
            logger.debug(f"[send_prompts_to_others] File {file_name} already processed, skip.")
            continue

        logger.debug(f"[send_prompts_to_others] Found file => {file_path}")

        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
            files = {'files': (os.path.basename(file_path), file_data, 'application/octet-stream')}
            logger.debug(f"[send_prompts_to_others] file={file_name}, size={len(file_data)} bytes read OK.")
        except Exception as e:
            logger.error(f"[send_prompts_to_others] Error reading file {file_path}: {e}")
            continue
        
        if receiver_url != sender_ip:
            payload = {'llm_anchor': llm_anchor}
            payload_json = json.dumps(payload)
            data = {'data': payload_json}

            logger.debug(f"[send_prompts_to_others] Sending {file_name} => http://{receiver_url}:30000/uploadfile")
            try:
                response = await post_files_with_retries(
                    f"http://{receiver_url}:30000/uploadfile",
                    files=files, data=data
                )
                if response is None:
                    logger.error(f"[send_prompts_to_others] No response sending {file_name} to {receiver_url}.")
                    continue
                if response.status_code == 200:
                    response_data = response.json()
                    responses.append(response_data)
                    logger.info(
                        f"[send_prompts_to_others] Sent file={file_name} to {receiver_url}, response={response_data}"
                    )
                else:
                    logger.error(
                        f"[send_prompts_to_others] Status={response.status_code} sending {file_name} to {receiver_url}"
                    )
            except Exception as ex:
                logger.error(
                    f"[send_prompts_to_others] Exception sending {file_name} => {receiver_url}: {ex}"
                )
            
            existing_record = await prompt_col.find_one({"file_name": file_name})
            if not existing_record:
                logger.debug(f"[send_prompts_to_others] after sending, can't find DB record for {file_name}.")
            else:
                deleted = await async_delete_with_retries(prompt_col, {"file_name": file_name})
                if not deleted:
                    logger.warning(f"[send_prompts_to_others] Could not delete doc for file_name={file_name}")
                else:
                    logger.info(f"[send_prompts_to_others] Deleted DB record for file_name={file_name}")

            processed_files.add(file_name)
            await async_force_delete_file(file_path)

        else:
            await prompt_col.update_one(
                {"file_name": file_name},
                {"$set": {"assigned": True}},
                upsert=True
            )
            local_process_mark = {
                "status": "success",
                "receive_time": None,
                "llm_anchor": llm_anchor,
                "slot_anchor": slot_anchor
            }
            responses.append(local_process_mark)
            logger.info(f"[send_prompts_to_others] Processed file {file_name} locally (assigned=True).")

    logger.debug(f"[send_prompts_to_others] Done. Collected responses: {responses}")
    return responses

class DataPayload(BaseModel):
    destinations: List[str]
    slot_anchor: int

@app.post("/process_data")
async def process_data(payload: DataPayload):
    if not db_ready or prompt_col is None or latency_col is None or local_ip_address is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    try:
        logger.debug(f"Processing data for slot_anchor={payload.slot_anchor}")
        destinations = payload.destinations
        slot_anchor = payload.slot_anchor

        responses = await send_prompts_to_others(destinations, slot_anchor, prompt_col, latency_col, local_ip_address) 

        processed_responses = set()
        all_responses_successful = False

        while not all_responses_successful:
            for response in responses:
                if response.get('status') == 'success' and json.dumps(response, sort_keys=True) not in processed_responses:
                    receive_time = response['receive_time']
                    slot_anchor_r = response['slot_anchor']
                    llm_anchor = response['llm_anchor']

                    if slot_anchor_r is None or llm_anchor is None:
                        processed_responses.add(json.dumps(response, sort_keys=True))
                        continue

                    query = {"slot_anchor": str(slot_anchor_r)}
                    projection = {f"{llm_anchor}.start": 1, "_id": 0}
                    start_data_point = await latency_col.find_one(query, projection)
                    start_time = start_data_point.get(llm_anchor, {}).get("start", None) if start_data_point else None

                    if receive_time is not None and start_time is not None:
                        uplink = receive_time - start_time
                        await latency_col.update_one(
                            {"slot_anchor": str(slot_anchor_r)},
                            {
                                "$set": {f"{str(llm_anchor)}.uplink": uplink,
                                         "slot_anchor": str(slot_anchor_r)},
                            },
                            upsert=True
                        )
                        logger.debug(f"Updated uplink latency for slot_anchor={slot_anchor_r}, llm_anchor={llm_anchor}")

                    processed_responses.add(json.dumps(response, sort_keys=True))

            all_responses_successful = (
                len(processed_responses) == len(responses) and 
                all(response.get('status') == 'success' for response in responses)
            )

            if not all_responses_successful:
                await asyncio.sleep(0.5)

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error in process_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_remotely")
async def inference_remotely(files: UploadFile = File(...), data: str = Form(...)):
    if not db_ready or latency_col is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    try:
        packaged_data = json.loads(data)
        inference_start = packaged_data.get('inference_start')
        inference_finish = packaged_data.get('inference_finish')
        slot_anchor = packaged_data.get('slot_anchor')
        llm_anchor = packaged_data.get('llm_anchor')

        downlink_finish = time.time()
        downlink = downlink_finish - inference_finish
        inference = inference_finish - inference_start

        projection = {"_id":0}
        for a in ANCHOR_LIST:
            projection[f"{a}.start"] = 1

        query = {"slot_anchor": str(slot_anchor)}
        doc = await latency_col.find_one(query, projection)
        if not doc:
            doc = {}

        start_time = doc.get(llm_anchor, {}).get("start", None) if llm_anchor in doc else None

        if start_time is None:
            await latency_col.update_one(
                {"slot_anchor": str(slot_anchor)},
                {
                    "$set": {
                        f"{llm_anchor}.skip": True,
                        f"{llm_anchor}.fail_reason": "no_start_time",
                        "slot_anchor": str(slot_anchor)
                    }
                },
                upsert=True
            )
            logger.info(f"[REMOTE] Inference skip: no start_time for slot={slot_anchor}, llm={llm_anchor}")
            return {"status": "skipped", "message": "No start_time found, skip recorded."}
        else:
            single_llm_delay = downlink_finish - start_time

            await latency_col.update_one(
                {"slot_anchor": str(slot_anchor)},
                {
                    "$set": {
                        f"{llm_anchor}.downlink": downlink,
                        f"{llm_anchor}.inference": inference,
                        f"{llm_anchor}.single_llm_delay": single_llm_delay,
                        "slot_anchor": str(slot_anchor)
                    },
                },
                upsert=True
            )

            logger.info(
                f"[REMOTE] Inference remotely recorded for slot_anchor={slot_anchor}, "
                f"llm_anchor={llm_anchor}, single_llm_delay={single_llm_delay}"
            )
            return {"status": "success", "message": "Latency recorded."}

    except Exception as e:
        logger.error(f"Error in inference_remotely: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class LocalPayload(BaseModel):
    slot_anchor: int
    llm_anchor: str
    inference_start: float
    inference_finish: float

@app.post("/inference_locally")
async def inference_locally(payload: LocalPayload, request: Request):
    if not db_ready or latency_col is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    try:
        client_ip = request.client.host
        logger.info(f"[LOCAL] Received from client_ip={client_ip}")
        inference_start = payload.inference_start
        inference_finish = payload.inference_finish
        slot_anchor = payload.slot_anchor
        llm_anchor = payload.llm_anchor

        projection = {"_id": 0}
        for a in ANCHOR_LIST:
            projection[f"{a}.start"] = 1

        query = {"slot_anchor": str(slot_anchor)}
        doc = await latency_col.find_one(query, projection)
        if not doc:
            doc = {}

        start_time = doc.get(llm_anchor, {}).get("start", None) if llm_anchor in doc else None

        if start_time is None:
            await latency_col.update_one(
                {"slot_anchor": str(slot_anchor)},
                {
                    "$set": {
                        f"{llm_anchor}.skip": True,
                        f"{llm_anchor}.fail_reason": "no_start_time",
                        "slot_anchor": str(slot_anchor)
                    }
                },
                upsert=True
            )
            logger.info(f"[LOCAL] Inference skip: no start_time for slot={slot_anchor}, llm={llm_anchor}")
            return {"status": "skipped", "message": "No start_time found, skip recorded."}
        else:
            inference_duration = inference_finish - inference_start
            single_llm_delay = inference_finish - start_time

            await latency_col.update_one(
                {"slot_anchor": str(slot_anchor)},
                {
                    "$set": {
                        f"{llm_anchor}.inference": inference_duration,
                        f"{llm_anchor}.single_llm_delay": single_llm_delay,
                        "slot_anchor": str(slot_anchor)
                    },
                },
                upsert=True
            )

            logger.info(
                f"[LOCAL] Inference locally recorded for slot_anchor={slot_anchor}, "
                f"llm_anchor={llm_anchor}, single_llm_delay={single_llm_delay}"
            )
            return {"status": "success", "message": "Latency recorded."}

    except Exception as e:
        logger.error(f"Error in inference_locally: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------        receive the execute command from the central server             ---------------------------------------
# ------------------------------        It will trigger the next training step                          ---------------------------------------
class TrainingContext:
    def __init__(self, server, agent, observe, explore):
        self.server = server
        self.agent = agent
        self.state = None
        # Post-action state: snapshot returned by server.step() *after* the
        # local agent's action has been applied. /get_local_state exposes this
        # so external calibration drivers can join backlog (Q_i at end of slot
        # t) with the deployment snapshot taken in the same slot, without the
        # time-skew that would arise from using the pre-action `self.state`.
        self.last_next_state = None
        self.last_slot_anchor = None
        self.last_state_ts = None
        self.index = 1
        self.initial_reset_done = False
        self.model_loaded = False
        self.state_dictionary = {}
        self.observe = observe
        self.explore = explore
        self.llm1_record = []
        self.llm2_record = []
        self.llm3_record = []
        self.llm4_record = []
    
    async def check_queue_age(self, slot_anchor, latency_col, max_age) -> bool:
        projection = {"_id": 0}
        for anchor_id in ANCHOR_LIST:
            projection[f"{anchor_id}.start"] = 1

        doc = await latency_col.find_one({"slot_anchor": str(slot_anchor)}, projection)
        if not doc:
            return False

        start_times = []
        for anchor_id in ANCHOR_LIST:
            st = doc.get(anchor_id, {}).get("start", None)
            if st is not None:
                start_times.append(st)

        if not start_times:
            return False
        earliest_start = min(start_times)
        queue_time = time.time() - earliest_start
        return (queue_time > max_age)
    
    async def cleanup_local_timeouts(self, max_age=900.0):
        """
        Scan /app/prompt_queue, parse each file, check if queue_time>max_age => remove file and doc.
        """
        import os, time, re
        prompt_path = "/app/prompt_queue"
        if not os.path.isdir(prompt_path):
            return  # no path => skip

        file_list = os.listdir(prompt_path)
        for filename in file_list:
            pattern = r"^(?P<slot_anchor>\d+)_(?P<sender_ip_underscore>(?:\d+_){3}\d+)_(?P<llm_anchor>\w+)_(?P<size>\d+)(?:\.txt|\.tar)$"
            parsed = re.match(pattern, filename)
            if not parsed:
                continue
            slot_anchor_str, sender_ip_underscore, llm_anchor, size = parsed.groups()
            slot_anchor_val = int(slot_anchor_str)

            # Look up the latency collection for the corresponding node.
            lat_col = self.server.mongo['latency'][sender_ip_underscore]
            is_timeout = await self.check_queue_age(slot_anchor_val, lat_col, max_age)
            if is_timeout:
                doc = await self.server.prompt_col.find_one({"file_name": filename})
                if doc and doc.get("be_processed") == True:
                    logger.info(f"[cleanup_local_timeouts] skip {filename}, because be_processed==True.")
                    continue
                
                # set fail_reason=timeout, single_llm_delay=900
                await lat_col.update_one(
                    {"slot_anchor": slot_anchor_str},
                    {
                        "$set": {
                            f"{llm_anchor}.fail_reason": "timeout",
                            f"{llm_anchor}.single_llm_delay": 900,
                        }
                    },
                    upsert=True
                )
                # remove local file
                full_path = os.path.join(prompt_path, filename)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"[cleanup_local_timeouts] removed {full_path} for slot={slot_anchor_val}, anchor={llm_anchor}")
                delete_result = await self.server.prompt_col.delete_one({"file_name": filename})
                if delete_result.deleted_count>0:
                    logger.info(f"[cleanup_local_timeouts] removed doc => {filename}")
                else:
                    logger.info(f"[cleanup_local_timeouts] no doc => {filename}")

    async def step(self, local_slot_anchor, status, incre_from):
        logger.debug(f"Training step called with local_slot_anchor={local_slot_anchor}, status={status}")
        if not self.initial_reset_done:
            self.server.reset()
            self.initial_reset_done = True

        self.state = await self.server.generate_state()
        self.last_slot_anchor = local_slot_anchor
        self.last_state_ts = time.time()
        logger.debug(f"self.state is {self.state}")

        mask = await self.server.compute_action_mask()
        # Use numpy boolean mask for actions; agents are written to accept
        # numpy arrays or list-like masks.
        action_mask = np.asarray(mask, dtype=bool)

        # Treat LyapunovAgent and GPTLocalAgent as direct controllers. They do
        # not use epsilon-greedy exploration inside TrainingContext.
        heur_like = (
            (LyapunovAgent is not None and isinstance(self.agent, LyapunovAgent))
            or isinstance(self.agent, GPTLocalAgent)
        )

        # f = math.exp(math.log(0.05 / 0.9) / self.explore)

        if local_slot_anchor - incre_from < self.observe:
            # Direct controllers choose from the current state and action mask.
            if heur_like:
                action = self.agent.choose_action(self.state, action_mask)
            else:
                valid_idx = np.nonzero(action_mask)[0]
                if len(valid_idx) == 0:
                    action = 0
                else:
                    action = int(valid_idx[np.random.randint(len(valid_idx))])
            self.state, next_state = await self.server.step(action)
            self.last_next_state = next_state
            self.last_state_ts = time.time()
            self.state_dictionary[local_slot_anchor] = [self.state, next_state, action]
            logger.debug(f"Observation step: action={action}, next_state={next_state}")

            reward_log = await self.server.reward_retrieval()
            for [local_sa, reward] in reward_log:
                [state_1, next_state_1, action_1] = self.state_dictionary[local_sa]
                reward_1 = reward
                del self.state_dictionary[local_sa]
                self.agent.memory.store_transition(state_1, action_1, reward_1, next_state_1)
                logger.debug(f"Storing transition in memory for slot_anchor={local_sa}, reward={reward_1}")

                data_reward = pd.DataFrame([[local_sa, reward]], columns=['local_slot_anchor', 'reward'])
                await asyncio.to_thread(
                    data_reward.to_csv,
                    '/app/results/data_reward.csv',
                    mode='a',
                    header=not os.path.exists('/app/results/data_reward.csv'),
                    index=False
                )

        else:
            # f = math.exp(math.log(0.05 / 0.9) / self.explore) # Exponential decay
            if not self.model_loaded and status == 'incre':
                self.agent.load_checkpoint()
                self.agent.load_replay_buffer("/app/results/replay_buffer.pkl")
                self.model_loaded = True
                logger.info("Model checkpoint loaded.")
            else:
                logger.info("Model train from scratch.")

            # Direct controllers bypass epsilon-greedy and choose a valid action.
            if heur_like:
                action = self.agent.choose_action(self.state, action_mask)
            else:
                eps_step = (0.9 - 0.05) / self.explore # Linear decay
                # self.agent.epsilon *= f
                # self.agent.epsilon = max(self.agent.epsilon, 0.05)
                self.agent.epsilon -= eps_step
                self.agent.epsilon = max(self.agent.epsilon, 0.05)
                if np.random.rand() < self.agent.epsilon:
                    valid_idx = np.nonzero(action_mask)[0]
                    if len(valid_idx) == 0:
                        action = 0
                    else:
                        action = int(valid_idx[np.random.randint(len(valid_idx))])
                else:
                    action = self.agent.choose_action(self.state, action_mask)
            self.state, next_state = await self.server.step(action)
            self.last_next_state = next_state
            self.last_state_ts = time.time()
            self.state_dictionary[local_slot_anchor] = [self.state, next_state, action]
            logger.debug(f"Training step: action={action}, next_state={next_state}")

            reward_log = await self.server.reward_retrieval()
            for [local_sa, reward] in reward_log:
                [state_1, next_state_1, action_1] = self.state_dictionary[local_sa]
                reward_1 = reward
                del self.state_dictionary[local_sa]
                self.agent.memory.store_transition(state_1, action_1, reward_1, next_state_1)
                logger.debug(f"Storing transition in memory for slot_anchor={local_sa}, reward={reward_1}")

                data_reward = pd.DataFrame([[local_sa, reward]], columns=['local_slot_anchor', 'reward'])
                await asyncio.to_thread(
                    data_reward.to_csv,
                    '/app/results/data_reward.csv',
                    mode='a',
                    header=not os.path.exists('/app/results/data_reward.csv'),
                    index=False
                )

            self.agent.learn()

            if local_slot_anchor % 2 == 0:
                self.agent.save_checkpoint()
                self.agent.save_replay_buffer("/app/results/replay_buffer.pkl")
                logger.info(f"Training step completed for local_slot_anchor={local_slot_anchor}. Checkpoint saved.")

        await self.cleanup_local_timeouts()

class ActionPayload(BaseModel):
    slot_anchor: int
    status: str
    incre_from: int
@app.post("/trigger")
async def trigger(payload: ActionPayload):
    global local_gpt_scheduler_task

    if not db_ready or training_context is None or server is None:
        raise HTTPException(status_code=400, detail="System not initialized yet.")

    logger.info(
        "Trigger called with slot_anchor=%s, status=%s, agent_class=%s",
        payload.slot_anchor,
        payload.status,
        getattr(getattr(training_context, "agent", None), "__class__", type("X", (), {})).__name__,
    )
    # Advance exactly one local step per external trigger for all agent
    # types (Lyapunov and GPTLocal), aligned to the
    # global slot cadence. We no longer start a separate background
    # scheduler for GPTLocalAgent so that higher-level controllers can
    # control when local decisions are made.
    local_slot_anchor = payload.slot_anchor // max(length_of_slot, 1)
    incre_from = payload.incre_from // max(length_of_slot, 1)

    await training_context.step(local_slot_anchor, payload.status, incre_from)

    # Snapshot the actual active LM set on this node after the local agent has
    # chosen its action and the environment state has been refreshed.
    # server.state[0:4] is pod_status (per-LM presence score:
    # >0 means LM type i+1 has at least one running pod on this node).
    global _active_lms_now
    try:
        st = getattr(server, "state", None)
        if st is not None and len(st) >= 4:
            new_active = [i + 1 for i, score in enumerate(st[:4]) if float(score) > 0.0]
        else:
            new_active = list(_active_lms_now)
        _active_lms_now = new_active
        if _runtime_event_logger is not None:
            _runtime_event_logger.log(
                outcome="success",
                slot_index=int(payload.slot_anchor),
                node=node_name or "",
                kind="lyapunov_active_set",
                active_lms=_active_lms_now,
                agent_mode=_current_agent_mode,
            )
    except Exception:
        # Telemetry must never break /trigger; swallow.
        pass

    return {"status": "success", "message": "Training step executed."}

# ------------------------------ Runtime reward export ------------------------------
@app.get("/download_reward")
async def download_reward():
    if not db_ready:
        raise HTTPException(status_code=400, detail="System not initialized yet.")
    reward_file_path = "/app/results/data_reward.csv"
    if not os.path.exists(reward_file_path):
        raise HTTPException(status_code=404, detail="No reward file found.")
    return FileResponse(path=reward_file_path, filename="data_reward.csv", media_type="text/csv")
