import os
import re
import asyncio
import aiofiles
import json
import csv
import time
import httpx
import tarfile
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

async def post_files_with_retries(url, files, data=None, max_retries=10, retry_delay=2, **kwargs):
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
# The main file handling function
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
    generates results into 'output_file',
    updates progress in 'progress_csv_path' after each line.
    But does *not* remove 'prompt_path' or clear progress here, 
    to allow safer resume if we crash mid-file.
    """
    # Check if the file is actually empty
    file_stat = await asyncio.to_thread(os.stat, prompt_path)
    if file_stat.st_size == 0:
        print(f"[handle_file] File {prompt_path} is zero bytes => skip inference.")
        return "SKIPPED"

    # Else proceed to read lines
    async with aiofiles.open(prompt_path, 'r') as f:
        sentences = await f.readlines()

    if not sentences:
        print(f"[handle_file] File {prompt_path} has no lines => skip inference.")
        return "SKIPPED"

    sentences = sentences[start_index:]
    async with aiofiles.open(output_file, 'a') as out_file:
        for idx, sentence in enumerate(sentences, start=start_index):
            sentence = sentence.strip()
            # generate
            inputs = await asyncio.to_thread(tokenizer, sentence, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = await asyncio.to_thread(model.generate, **inputs, max_length=50, num_return_sequences=1)
            generated_text = await asyncio.to_thread(tokenizer.decode, outputs[0], skip_special_tokens=True)
            await out_file.write(generated_text + "\n")

            # save progress after each line
            await save_progress(progress_csv_path, os.path.basename(prompt_path), idx + 1, inference_start)

    print(f"[handle_file] Done reading file {prompt_path}, but not removing yet.")
    return "DONE"


async def finalize_file(
    prompt_path,
    progress_csv_path,
    prompt_col=None,
    filename=None
):
    """
    Once we've *confirmed* the entire process (including sending results) 
    is successfully done, then remove the local prompt file 
    and clear the progress CSV.
    """
    # 1) Clear the progress
    await clear_progress(progress_csv_path)
    # 2) Remove the file
    if os.path.exists(prompt_path):
        await asyncio.to_thread(os.remove, prompt_path)
        print(f"[finalize_file] Removed prompt file: {prompt_path}")
    else:
        print(f"[finalize_file] prompt file not found (already removed?): {prompt_path}")

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
async def send_inference_result(
    sender_ip, 
    local_ip, 
    slot_anchor, 
    llm_anchor, 
    inference_start, 
    inference_finish, 
    output_file
):
    """
    If the file was truly processed (non-empty), 
    we call local/remote inference API so that remote can record the latency.
    If you'd prefer skipping for zero-length or error, 
    do not call this method at all.
    """
    print(f"Preparing to send inference result for slot_anchor={slot_anchor}, llm_anchor={llm_anchor}")
    if sender_ip == local_ip:
        # local case
        url = f"http://{sender_ip}:30000/inference_locally"
        payload = {
            'slot_anchor': slot_anchor,
            'llm_anchor': llm_anchor,
            'inference_start': inference_start,
            'inference_finish': inference_finish,
        }
        while True:
            response = await post_with_retries(url, json=payload)
            if response and response.status_code == 200:
                print(f"[LOCAL] Inference result successfully sent for slot={slot_anchor}, removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[LOCAL] Failed to send or not acknowledged. Will retry in 1s... {response}")
                await asyncio.sleep(1)
    else:
        # remote case
        url = f"http://{sender_ip}:30000/inference_remotely"
        downlink_start = time.time()
        payload = {
            'slot_anchor': slot_anchor,
            'llm_anchor': llm_anchor,
            'inference_start': inference_start,
            'inference_finish': inference_finish,
            'downlink_start': downlink_start
        }
        payload_json = json.dumps(payload)
        async with aiofiles.open(output_file, 'rb') as f:
            file_data = await f.read()
        files = {'files': ('filename', file_data, 'application/octet-stream')}
        data = {'data': payload_json}

        while True:
            response = await post_files_with_retries(url, files=files, data=data)
            if response and response.status_code == 200:
                print(f"[REMOTE] Inference result successfully sent for slot={slot_anchor}, removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[REMOTE] Failed to send or not acknowledged. Will retry in 10s... {response}")
                await asyncio.sleep(1)


###############################
# main
###############################
async def main():
    if is_running_in_kubernetes():
        llm_anchor = 1
        prompt_father_path = '/app/prompt_queue'
        gpu_usage = os.getenv("USE_GPU")
        local_ip = os.getenv('NODE_IP')

        # 1) K8s + Mongo init
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
            
        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")
        prompt_col = mongo['prompt_recorder'][f"{local_ip.replace('.', '_')}"]

        # 2) Model init
        tokenizer = AutoTokenizer.from_pretrained('/app/tokenizer', use_fast=True)
        if gpu_usage != "0":
            model = AutoModelForCausalLM.from_pretrained('/app/gpt2_model',
                                                         device_map='auto',
                                                         low_cpu_mem_usage=True)
        else:
            model = AutoModelForCausalLM.from_pretrained('/app/gpt2_model',
                                                         low_cpu_mem_usage=True)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        # 3) Resume
        current_processing_prompt_path = '/app/current_prompt'
        processing_status = f"{current_processing_prompt_path}/current_prompt_{llm_anchor}.csv"
        last_file, last_index, inference_start = await load_progress(processing_status)
        if last_file:
            # check file existence
            prompt_path_maybe = os.path.join(prompt_father_path, last_file)
            if not os.path.exists(prompt_path_maybe):
                print(f"[RESUME] Found saved progress but file is missing => clearing progress.")
                await clear_progress(processing_status)
            else:
                print(f"[RESUME] from last progress: {last_file}, index={last_index}, start={inference_start}")
                slot_anchor, sender_ip, llm_anch = parse_prompt_filename(last_file)
                if sender_ip == local_ip:
                    output_file = os.path.join('/app/output_local',
                                               f"{slot_anchor}_{llm_anch}.txt")
                else:
                    output_file = os.path.join('/app/output_relay',
                                               f"{slot_anchor}_{llm_anch}.txt")

                # resume handle
                result_of_file = await handle_file(prompt_path_maybe,
                                                  output_file,
                                                  model,
                                                  tokenizer,
                                                  processing_status,
                                                  inference_start,
                                                  start_index=last_index)
                if result_of_file == "DONE":
                    inference_finish = time.time()
                    await send_inference_result(sender_ip,
                                                local_ip,
                                                slot_anchor,
                                                llm_anch,
                                                inference_start,
                                                inference_finish,
                                                output_file)
                    await finalize_file(prompt_path_maybe, processing_status, prompt_col=prompt_col, filename=last_file)
                else:
                    # means SKIPPED => this will neve happen. hande_file will not save_progress if it skips.
                    await asyncio.to_thread(os.remove, prompt_path_maybe)
                    await clear_progress(processing_status)

        # 4) infinite loop
        while True:
            file_info = sorting(prompt_father_path, llm_anchor)
            if not file_info:
                print("[INFO] No new files. Waiting for new tasks...")
                await asyncio.sleep(5)
                continue

            for slot_anchor, sender_ip, llm_anch, filename in file_info:
                # (4.1) check if assigned
                existing_record = await prompt_col.find_one({"file_name": filename})
                if not existing_record or not existing_record.get("assigned"):
                    print(f"File {filename} not assigned to {local_ip}, skip.")
                    continue

                await prompt_col.update_one(
                    {"file_name": filename},
                    {"$set": {"be_processed": True}}
                )

                # (4.2) local or remote out path
                prompt_path_full = os.path.join(prompt_father_path, filename)
                if sender_ip == local_ip:
                    output_file = os.path.join('/app/output_local',
                                               f"{slot_anchor}_{llm_anch}.txt")
                else:
                    output_file = os.path.join('/app/output_relay',
                                               f"{slot_anchor}_{llm_anch}.txt")

                # (4.4) do handle
                inference_start = time.time()
                print(f"Processing new file: {filename}, slot={slot_anchor}, anchor={llm_anch}")
                result_of_file = await handle_file(prompt_path_full,
                                                  output_file,
                                                  model,
                                                  tokenizer,
                                                  processing_status,
                                                  inference_start,
                                                  start_index=0)
                if result_of_file == "DONE":
                    # send result
                    inference_finish = time.time()
                    await send_inference_result(sender_ip,
                                                local_ip,
                                                slot_anchor,
                                                llm_anch,
                                                inference_start,
                                                inference_finish,
                                                output_file)
                    # finalize
                    await finalize_file(prompt_path_full, processing_status, prompt_col=prompt_col, filename=filename)
                else:
                    # means "SKIPPED" => e.g. file had zero lines or zero bytes
                    # remove local prompt, no inference call.
                    if os.path.exists(prompt_path_full):
                        await asyncio.to_thread(os.remove, prompt_path_full)
                    await clear_progress(processing_status)
                    
                    print(f"[SKIPPED] Removed zero-size file {prompt_path_full} with no inference call.")

            print("[INFO] All inference tasks completed for current batch. Will continue waiting for new files...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
