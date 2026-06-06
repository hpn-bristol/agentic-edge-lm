import os
import re
import asyncio
import aiofiles
import json
import csv
import gc
import time
import torch
import httpx
from kubernetes import client, config
from motor.motor_asyncio import AsyncIOMotorClient   
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

###############################
# asynchronise http requests
###############################
async def post_with_retries(url, json=None, max_retries=10, retry_delay=2, **kwargs):
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=json, **kwargs)
                response.raise_for_status()
                return response
            except httpx.RequestError as e:
                print(f"[{url}] Attempt {attempt + 1}/{max_retries} failed with error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"All {max_retries} attempts failed. Returning None.")
                    return None

async def post_files_with_retries(url, files, data=None, max_retries=10, retry_delay=1, **kwargs):
    async with httpx.AsyncClient(verify=False) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, files=files, data=data, **kwargs)
                response.raise_for_status()
                return response
            except httpx.RequestError as e:
                print(f"[{url}] Attempt {attempt + 1}/{max_retries} to {url} failed with error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"All {max_retries} attempts failed. Returning None.")
                    return None

###############################
# tool functions
###############################
def parse_prompt_filename(filename):
    pattern = r"(?P<slot_anchor>\d+)_(?P<sender_ip_underscore>(?:\d+_){3}\d+)_(?P<llm_anchor>\w+)_(?P<size>\d+)"
    match = re.match(pattern, filename)
    if match:
        slot_anchor = int(match.group("slot_anchor"))
        sender_ip_underscore = match.group("sender_ip_underscore")
        sender_ip = sender_ip_underscore.replace("_", ".")
        llm_anchor = match.group("llm_anchor")
        return slot_anchor, sender_ip, llm_anchor
    return None, None, None

def is_running_in_kubernetes():
    return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount")

def sorting(prompt_path, allocate_llm):
    files_info = []
    for filename in os.listdir(prompt_path):
        parsed = parse_prompt_filename(filename)
        if parsed:
            slot_anchor, sender_ip, llm = parsed
            if llm == str(allocate_llm):
                files_info.append((slot_anchor, sender_ip, llm, filename))
    files_info.sort(key=lambda x: int(x[0]))
    return files_info

async def save_progress(progress_csv_path, filename, index, inference_start):
    def sync_write():
        with open(progress_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename, index, inference_start])
    await asyncio.to_thread(sync_write)

async def load_progress(progress_csv_path):
    if not await asyncio.to_thread(os.path.exists, progress_csv_path):
        return None, 0, 0.0
    def sync_read():
        with open(progress_csv_path, mode='r') as file:
            reader = csv.reader(file)
            row = next(reader, None)
            if row:
                return row[0], int(row[1]), float(row[2])
            return None, 0, 0.0
    return await asyncio.to_thread(sync_read)

async def clear_progress(progress_csv_path):
    def sync_clear():
        with open(progress_csv_path, mode='w', newline='') as file:
            file.truncate()
    await asyncio.to_thread(sync_clear)


###############################
# handle_file + finalize_file
###############################
async def handle_file(
    prompt_path, 
    output_file, 
    model, 
    tokenizer, 
    progress_csv_path, 
    inference_start, 
    start_index=0
):
    """
    Reads lines from 'prompt_path' starting from 'start_index',
    does inference, writes to 'output_file', and updates progress CSV each line.
    Returns "DONE" or "SKIPPED" to indicate if we successfully processed it or not.
    """
    # 1) Check if file is zero bytes
    stat_result = await asyncio.to_thread(os.stat, prompt_path)
    if stat_result.st_size == 0:
        print(f"[handle_file] {prompt_path} is zero bytes => skip.")
        return "SKIPPED"

    # 2) Read lines
    async with aiofiles.open(prompt_path, 'r') as f:
        lines = await f.readlines()
    if not lines:
        print(f"[handle_file] {prompt_path} has no lines => skip.")
        return "SKIPPED"

    # 3) Inference line by line
    lines = lines[start_index:]
    async with aiofiles.open(output_file, 'a') as out_file:
        for idx, sentence in enumerate(lines, start=start_index):
            sentence = sentence.strip()
            # do inference
            inputs = await asyncio.to_thread(tokenizer, sentence, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = await asyncio.to_thread(model.generate, **inputs, max_length=50, num_return_sequences=1)
            generated_text = await asyncio.to_thread(tokenizer.decode, outputs[0], skip_special_tokens=True)
            # write out
            await out_file.write(generated_text + "\n")

            # save progress
            await save_progress(progress_csv_path, os.path.basename(prompt_path), idx + 1, inference_start)

    print(f"[handle_file] Completed reading {prompt_path}, but not removing it yet.")
    return "DONE"


async def finalize_file(prompt_path, progress_csv_path, prompt_col=None, filename=None):
    """
    After we're sure all done (including send_inference_result), remove the prompt
    and clear CSV so we won't resume next time.
    """
    # 1) Clear CSV
    await clear_progress(progress_csv_path)
    # 2) Remove prompt file
    if os.path.exists(prompt_path):
        await asyncio.to_thread(os.remove, prompt_path)
        print(f"[finalize_file] Removed file: {prompt_path}")
    else:
        print(f"[finalize_file] Already removed: {prompt_path}")
    
    # 3) Remove doc from prompt_col (if provided)
    if prompt_col is not None and filename:
        result = await prompt_col.delete_one({"file_name": filename})
        if result.deleted_count > 0:
            print(f"[finalize_file] Removed prompt_col record for file_name={filename}")
        else:
            print(f"[finalize_file] No doc found in prompt_col with file_name={filename}. Possibly already removed.")

###############################
# send_inference_result
###############################
async def send_inference_result(sender_ip, local_ip, slot_anchor, llm_anchor, inference_start, inference_finish, output_file):
    """
    If "DONE", we call this to let the remote or local know about inference result.
    If "SKIPPED", we skip calling this.
    """
    print(f"Preparing to send inference result for slot={slot_anchor}, llm={llm_anchor}")
    if sender_ip == local_ip:
        # local
        url = f"http://{sender_ip}:30000/inference_locally"
        payload = {
            "slot_anchor": slot_anchor,
            "llm_anchor": llm_anchor,
            "inference_start": inference_start,
            "inference_finish": inference_finish
        }
        while True:
            resp = await post_with_retries(url, json=payload)
            if resp and resp.status_code == 200:
                print(f"[LOCAL] inference result success => removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[LOCAL] fail or not ack. retry in 10s... {resp}")
                await asyncio.sleep(1)
    else:
        # remote
        url = f"http://{sender_ip}:30000/inference_remotely"
        downlink_start = time.time()
        payload = {
            "slot_anchor": slot_anchor,
            "llm_anchor": llm_anchor,
            "inference_start": inference_start,
            "inference_finish": inference_finish,
            "downlink_start": downlink_start
        }
        p_json = json.dumps(payload)
        async with aiofiles.open(output_file, 'rb') as f:
            file_data = await f.read()
        files = {"files": ("filename", file_data, "application/octet-stream")}
        data = {"data": p_json}

        while True:
            resp = await post_files_with_retries(url, files=files, data=data)
            if resp and resp.status_code == 200:
                print(f"[REMOTE] inference result success => removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[REMOTE] fail or not ack. retry in 1s... {resp}")
                await asyncio.sleep(1)

###############################
# main
###############################
async def main():
    if is_running_in_kubernetes():
        llm_anchor = 2
        prompt_path = "/app/prompt_queue"
        gpu_usage = os.getenv("USE_GPU")
        local_ip = os.getenv("NODE_IP")

        # 1) K8s / Mongo init
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")

        prompt_col = mongo["prompt_recorder"][f"{local_ip.replace('.', '_')}"]

        # 2) Model init
        tokenizer = AutoTokenizer.from_pretrained("/app/tokenizer_large", use_fast=True)
        if gpu_usage != "0":
            model = AutoModelForCausalLM.from_pretrained(
                "/app/gpt2_large_model",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "/app/gpt2_large_model",
                low_cpu_mem_usage=True
            )

        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        # CSV for resume
        current_dir = "/app/current_prompt"
        progress_csv = os.path.join(current_dir, f"current_prompt_{llm_anchor}.csv")

        # 3) Attempt resume
        last_file, last_idx, inf_start = await load_progress(progress_csv)
        if last_file:
            # check if it still exist
            prompt_full = os.path.join(prompt_path, last_file)
            if not os.path.exists(prompt_full):
                print(f"[RESUME] Saved progress has no matching file => clearing progress.")
                await clear_progress(progress_csv)
            else:
                print(f"[RESUME] from {last_file}, idx={last_idx}")
                slot_a, s_ip, l_anchor = parse_prompt_filename(last_file)
                if s_ip == local_ip:
                    out_file = os.path.join("/app/output_local", f"{slot_a}_{l_anchor}.txt")
                else:
                    out_file = os.path.join("/app/output_relay", f"{slot_a}_{l_anchor}.txt")

                # do handle
                status = await handle_file(prompt_full, out_file, model,
                                           tokenizer, progress_csv, inf_start,
                                           start_index=last_idx)
                if status == "DONE":
                    # send result
                    t_finish = time.time()
                    await send_inference_result(s_ip, local_ip, slot_a, l_anchor,
                                                inf_start, t_finish, out_file)
                    # finalize
                    await finalize_file(prompt_full, progress_csv, prompt_col, last_file)
                else:
                    # "SKIPPED" => remove prompt & CSV
                    await asyncio.to_thread(os.remove, prompt_full)
                    await clear_progress(progress_csv)

        # 4) infinite
        while True:
            files_info = sorting(prompt_path, llm_anchor)
            if not files_info:
                print("[INFO] No new files. Sleep 5s..")
                await asyncio.sleep(5)
                continue

            for slot_a, s_ip, l_anchor, fname in files_info:
                rec = await prompt_col.find_one({"file_name": fname})
                if not rec or not rec.get("assigned"):
                    print(f"File {fname} not assigned => skip.")
                    continue

                await prompt_col.update_one(
                    {"file_name": fname},
                    {"$set": {"be_processed": True}}
                )

                full_p = os.path.join(prompt_path, fname)

                # decide out file
                if s_ip == local_ip:
                    out_file = os.path.join("/app/output_local", f"{slot_a}_{l_anchor}.txt")
                else:
                    out_file = os.path.join("/app/output_relay", f"{slot_a}_{l_anchor}.txt")

                inf_start2 = time.time()
                print(f"Processing {fname}, anchor={slot_a}, l_anchor={l_anchor}")
                ret = await handle_file(full_p, out_file, model, tokenizer,
                                        progress_csv, inf_start2, start_index=0)
                if ret == "DONE":
                    # do inference result
                    inf_finish2 = time.time()
                    await send_inference_result(s_ip, local_ip, slot_a, l_anchor,
                                                inf_start2, inf_finish2, out_file)
                    await finalize_file(full_p, progress_csv, prompt_col, fname)
                else:
                    # "SKIPPED"
                    if os.path.exists(full_p):
                        await asyncio.to_thread(os.remove, full_p)
                    await clear_progress(progress_csv)

                    print(f"[SKIPPED] Removed zero-size (or empty) {full_p} no inference call.")

            print("[INFO] All tasks done for now.")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
