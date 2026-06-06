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
import tarfile
import io

from kubernetes import client, config
from motor.motor_asyncio import AsyncIOMotorClient
from diffusers import DiffusionPipeline


###############################
# Retry utilities
###############################
async def post_with_retries(url, json=None, max_retries=10, retry_delay=0.5, **kwargs):
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=json, **kwargs)
                response.raise_for_status()
                return response
            except httpx.RequestError as e:
                print(f"[{url}] Attempt {attempt+1}/{max_retries} failed with error: {e}")
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
                print(f"[{url}] Attempt {attempt+1}/{max_retries} to {url} failed with error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"All {max_retries} attempts failed. Returning None.")
                    return None


###############################
# Tool / Helper Functions
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

def is_running_in_kubernetes():
    return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount")

###############################
# finalize_file
###############################
async def finalize_file(prompt_path, progress_csv_path, prompt_col=None, filename=None):
    """
    After we've fully processed + sent the result, remove the text prompt 
    and clear progress CSV.
    """
    # 1) Clear progress
    await clear_progress(progress_csv_path)

    # 2) Remove prompt if it exists
    if os.path.exists(prompt_path):
        await asyncio.to_thread(os.remove, prompt_path)
        print(f"[finalize_file] Removed prompt {prompt_path}")
    else:
        print(f"[finalize_file] Prompt file not found => {prompt_path}")
    if prompt_col is not None and filename:
        result = await prompt_col.delete_one({"file_name": filename})
        if result.deleted_count > 0:
            print(f"[finalize_file] Removed record in prompt_col for file_name={filename}")
        else:
            print(f"[finalize_file] No prompt_col doc with file_name={filename}, maybe already removed.")



###############################
# handle_file
###############################
async def handle_file(
    prompt_path,
    output_file,
    pipeline,
    progress_csv_path,
    inference_start,
    start_index=0
):
    """
    Reads lines from 'prompt_path' and runs stable-diffusion pipeline 
    -> writes images to 'output_file' (tar).
    If file is zero or unreadable, we skip.
    """

    # 1) If file is zero bytes => skip
    stat_info = await asyncio.to_thread(os.stat, prompt_path)
    if stat_info.st_size == 0:
        print(f"[handle_file] {prompt_path} is zero bytes => skip.")
        return "SKIPPED"

    # 2) read lines
    def sync_read_sentences():
        with open(prompt_path, 'r') as f:
            return f.readlines()
    try:
        lines = await asyncio.to_thread(sync_read_sentences)
    except (FileNotFoundError, OSError) as e:
        print(f"[handle_file] read error => skip. {e}")
        return "SKIPPED"

    if not lines:
        print(f"[handle_file] {prompt_path} has no lines => skip.")
        return "SKIPPED"

    # 3) Determine tar mode
    def sync_determine_mode():
        if os.path.exists(output_file):
            return "a"
        else:
            return "w"
    mode = await asyncio.to_thread(sync_determine_mode)

    idx = start_index - 1

    async def process_line(sentence, i):
        sentence = sentence.strip()
        # call pipeline
        images = await asyncio.to_thread(pipeline, sentence)
        image = images.images[0]

        # save in-memory
        buffer = io.BytesIO()
        await asyncio.to_thread(image.save, buffer, format="PNG")
        buffer.seek(0)

        # create TarInfo
        tar_info = tarfile.TarInfo(name=f"image_{i}.png")
        tar_info.size = buffer.getbuffer().nbytes
        return (tar_info, buffer)

    tar_entries = []
    for i, line in enumerate(lines[start_index:], start=start_index):
        tar_info, buf = await process_line(line, i)
        tar_entries.append((tar_info, buf))
        idx = i
        # save partial progress
        await save_progress(progress_csv_path, os.path.basename(prompt_path), idx+1, inference_start)

    # 4) Actually write them out to the tar
    def sync_write_tar():
        with tarfile.open(output_file, mode) as tar:
            for info, buf in tar_entries:
                tar.addfile(info, fileobj=buf)

    if tar_entries:
        await asyncio.to_thread(sync_write_tar)

    print(f"[handle_file] Completed reading {prompt_path}, but not removing or clearing CSV.")
    return "DONE"


###############################
# send_inference_result
###############################
async def send_inference_result(sender_ip, local_ip, slot_anchor, llm_anchor,
                                inference_start, inference_finish, output_file):
    print(f"[INFO] send_inference_result: slot={slot_anchor}, anchor={llm_anchor}, output={output_file}")

    if sender_ip == local_ip:
        # local
        url = f"http://{sender_ip}:30000/inference_locally"
        payload = {
            'slot_anchor': slot_anchor,
            'llm_anchor': llm_anchor,
            'inference_start': inference_start,
            'inference_finish': inference_finish
        }
        while True:
            resp = await post_with_retries(url, json=payload)
            if resp and resp.status_code == 200:
                print(f"[LOCAL] Done => remove {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[LOCAL] Not ack => sleep 1s. resp={resp}")
                await asyncio.sleep(1)

    else:
        # remote
        url = f"http://{sender_ip}:30000/inference_remotely"
        payload = {
            'slot_anchor': slot_anchor,
            'llm_anchor': llm_anchor,
            'inference_start': inference_start,
            'inference_finish': inference_finish,
            'downlink_start': time.time()
        }
        p_json = json.dumps(payload)

        def sync_read_tar():
            with open(output_file, 'rb') as f:
                return f.read()
        tar_data = await asyncio.to_thread(sync_read_tar)

        files = {'files': ('output.tar', tar_data, 'application/x-tar')}
        data = {'data': p_json}

        while True:
            resp = await post_files_with_retries(url, files=files, data=data)
            if resp and resp.status_code == 200:
                print(f"[REMOTE] Done => remove {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[REMOTE] Not ack => sleep 1s. resp={resp}")
                await asyncio.sleep(1)


###############################
# main
###############################
async def main():
    if is_running_in_kubernetes():
        llm_anchor = 4
        prompt_father_path = '/app/prompt_queue'
        gpu_usage = os.getenv("USE_GPU")
        local_ip = os.getenv("NODE_IP")

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")
        prompt_col = mongo['prompt_recorder'][f"{local_ip.replace('.', '_')}"]

        # load pipeline
        if gpu_usage == "0":
            pipeline = DiffusionPipeline.from_pretrained(
                "/app/stable-diffusion-v1-5",
                low_cpu_mem_usage=True
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                "/app/stable-diffusion-v1-5",
                low_cpu_mem_usage=True
            )
            pipeline.to("cuda")

        current_processing_prompt_path = '/app/current_prompt'
        progress_csv_path = os.path.join(current_processing_prompt_path, f"current_prompt_{llm_anchor}.csv")

        # 1) Attempt resume
        old_file, old_idx, old_start = await load_progress(progress_csv_path)
        if old_file:
            old_path = os.path.join(prompt_father_path, old_file)
            if not os.path.exists(old_path):
                print(f"[RESUME] Saved file is missing => clearing progress.")
                await clear_progress(progress_csv_path)
            else:
                slot_anchor, sender_ip, llm_anch = parse_prompt_filename(old_file)
                if sender_ip == local_ip:
                    output_file = os.path.join('/app/output_local', f"{slot_anchor}_{llm_anch}.tar")
                else:
                    output_file = os.path.join('/app/output_relay', f"{slot_anchor}_{llm_anch}.tar")

                print(f"[RESUME] from {old_file}, idx={old_idx}, start={old_start}")
                result = await handle_file(old_path, output_file, pipeline, progress_csv_path, float(old_start), start_index=old_idx)
                if result == "DONE":
                    finish_time = time.time()
                    await send_inference_result(sender_ip, local_ip, slot_anchor, llm_anch,
                                                float(old_start), finish_time, output_file)
                    await finalize_file(old_path, progress_csv_path, prompt_col, old_file)
                else:
                    # "SKIPPED"
                    if os.path.exists(old_path):
                        await asyncio.to_thread(os.remove, old_path)
                    await clear_progress(progress_csv_path)
                    print(f"[RESUME] SKIPPED => removed {old_path}")

        # 2) infinite loop
        while True:
            file_info = sorting(prompt_father_path, llm_anchor)
            if not file_info:
                print("[INFO] no new text => sleep 5s..")
                await asyncio.sleep(5)
                continue

            for slot_anchor, sender_ip, llm_anch, filename in file_info:
                # assigned check
                rec = await prompt_col.find_one({"file_name": filename})
                if not rec or not rec.get("assigned"):
                    print(f"File {filename} not assigned => skip..")
                    continue

                await prompt_col.update_one(
                    {"file_name": filename},
                    {"$set": {"be_processed": True}}
                )

                # local or remote
                prompt_path = os.path.join(prompt_father_path, filename)
                if sender_ip == local_ip:
                    output_tar = os.path.join('/app/output_local', f"{slot_anchor}_{llm_anch}.tar")
                else:
                    output_tar = os.path.join('/app/output_relay', f"{slot_anchor}_{llm_anch}.tar")

                # do inference
                start_t = time.time()
                print(f"[INFO] handling file={filename}, slot={slot_anchor}, anchor={llm_anch}")
                result = await handle_file(prompt_path, output_tar, pipeline, progress_csv_path, start_t, start_index=0)
                if result == "DONE":
                    finish_t = time.time()
                    await send_inference_result(sender_ip, local_ip, slot_anchor, llm_anch, start_t, finish_t, output_tar)
                    await finalize_file(prompt_path, progress_csv_path, prompt_col, filename)
                else:
                    # "SKIPPED"
                    if os.path.exists(prompt_path):
                        await asyncio.to_thread(os.remove, prompt_path)
                    await clear_progress(progress_csv_path)
                    print(f"[SKIP] zero/empty => removed {prompt_path}")

            print("[INFO] done this batch => sleep 5s..")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
