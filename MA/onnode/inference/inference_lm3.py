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
from PIL import Image
import io
from kubernetes import client, config
from motor.motor_asyncio import AsyncIOMotorClient   
from transformers import BlipProcessor, BlipForConditionalGeneration


###############################
# asynchronise http requests
###############################
async def post_with_retries(url, json=None, max_retries=10, retry_delay=0.5, **kwargs):
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

###############################
# finalize_file
###############################
async def finalize_file(prompt_path, progress_csv_path, prompt_col=None, filename=None):
    """
    Removes the tar file & clears progress after success
    """
    # clear CSV
    await clear_progress(progress_csv_path)
    # remove tar
    if os.path.exists(prompt_path):
        await asyncio.to_thread(os.remove, prompt_path)
        print(f"[finalize_file] Removed: {prompt_path}")
    else:
        print(f"[finalize_file] Already removed or missing: {prompt_path}")
    
    if prompt_col is not None and filename:
        result = await prompt_col.delete_one({"file_name": filename})
        if result.deleted_count > 0:
            print(f"[finalize_file] Removed record in prompt_col for file_name={filename}")
        else:
            print(f"[finalize_file] No prompt_col doc with file_name={filename}, maybe already removed.")


###############################
# handle_tar_file
###############################
async def handle_tar_file(
    prompt_path, 
    output_file, 
    model, 
    processor, 
    progress_csv_path, 
    inference_start, 
    start_index=0
):
    """
    Opens tar, reads members from start_index onward. 
    Writes results to output_file. 
    Updates progress after each member. 
    Returns 'DONE' or 'SKIPPED'.
    """
    # 1) If tar is zero bytes => skip
    stat_info = await asyncio.to_thread(os.stat, prompt_path)
    if stat_info.st_size == 0:
        print(f"[handle_tar_file] {prompt_path} is zero bytes => skip.")
        return "SKIPPED"

    # 2) get tar members
    def sync_tar_members():
        try:
            with tarfile.open(prompt_path, 'r') as tar:
                members = tar.getmembers()
            return members
        except (tarfile.ReadError, FileNotFoundError) as e:
            print(f"[ERROR] tar reading {prompt_path}: {e}")
            return []
    members = await asyncio.to_thread(sync_tar_members)
    if not members:
        print(f"[handle_tar_file] {prompt_path} has no members => skip.")
        return "SKIPPED"

    # skip ones we've done
    members = members[start_index:]
    idx = start_index - 1

    async def process_member(member, idx):
        def sync_extract():
            with tarfile.open(prompt_path, 'r') as tar:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    image_data = extracted.read()
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    return image
            return None

        image = await asyncio.to_thread(sync_extract)
        if image is None:
            return idx  # skip broken file

        # do inference
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # append to output
        async with aiofiles.open(output_file, 'a') as f:
            await f.write(f"{member.name}: {caption}\n")

        idx += 1
        # record progress
        await save_progress(progress_csv_path, os.path.basename(prompt_path), idx+1, inference_start)
        return idx

    # 3) loop through members
    for member in members:
        if member.isfile():
            idx = await process_member(member, idx)

    print(f"[handle_tar_file] Completed reading tar={prompt_path}. No final remove or CSV clear here.")
    return "DONE"


###############################
# send_inference_result
###############################
async def send_inference_result(sender_ip, local_ip, slot_anchor, llm_anchor, 
                                inference_start, inference_finish, output_file):
    # same as your existing code, plus a small change to skip if 'SKIPPED'
    # but we'll do that check externally in 'main'
    print(f"[send_inference_result] about to post to local/remote. slot={slot_anchor} anchor={llm_anchor}")

    if sender_ip == local_ip:
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
                print(f"[LOCAL] Sent result, removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[LOCAL] Not ack: {resp}, try in 1s.")
                await asyncio.sleep(1)
    else:
        url = f"http://{sender_ip}:30000/inference_remotely"
        payload = {
            'slot_anchor': slot_anchor,
            'llm_anchor': llm_anchor,
            'inference_start': inference_start,
            'inference_finish': inference_finish,
            'downlink_start': time.time()
        }
        pay_json = json.dumps(payload)
        async with aiofiles.open(output_file, 'rb') as f:
            f_data = await f.read()
        files = {'files': ('filename', f_data, 'application/octet-stream')}
        data = {'data': pay_json}
        while True:
            resp = await post_files_with_retries(url, files=files, data=data)
            if resp and resp.status_code == 200:
                print(f"[REMOTE] Sent result, removing {output_file}")
                await asyncio.to_thread(os.remove, output_file)
                break
            else:
                print(f"[REMOTE] Not ack: {resp}, try in 1s.")
                await asyncio.sleep(1)


###############################
# main
###############################
async def main():
    if is_running_in_kubernetes():
        llm_anchor = 3
        prompt_father_path = '/app/prompt_queue'
        gpu_usage = os.getenv("USE_GPU")
        local_ip = os.getenv('NODE_IP')

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(name="mongodb-service", namespace="mongodb")
        mongo = AsyncIOMotorClient(f"mongodb://root:secret@{service.spec.cluster_ip}:27017")
        prompt_col = mongo['prompt_recorder'][f"{local_ip.replace('.', '_')}"]

        processor = BlipProcessor.from_pretrained("/app/blip-processor")
        if gpu_usage != "0":
            model = BlipForConditionalGeneration.from_pretrained("/app/blip-model", low_cpu_mem_usage=True).to("cuda")
        else:
            model = BlipForConditionalGeneration.from_pretrained("/app/blip-model", low_cpu_mem_usage=True)

        if model.config.pad_token_id is None and model.config.eos_token_id is not None:
            model.config.pad_token_id = model.config.eos_token_id

        current_processing_prompt_path = '/app/current_prompt'
        progress_csv_path = f"{current_processing_prompt_path}/current_prompt_{llm_anchor}.csv"

        # 1) resume if needed
        last_file, last_index, inference_start = await load_progress(progress_csv_path)
        if last_file:
            # see if that tar still exist
            tar_path_resume = os.path.join(prompt_father_path, last_file)
            if not os.path.exists(tar_path_resume):
                print(f"[RESUME] Saved tar is missing => clearing progress.")
                await clear_progress(progress_csv_path)
            else:
                # parse
                slot_anchor, sender_ip, llm_anch = parse_prompt_filename(last_file)
                if sender_ip == local_ip:
                    output_file = os.path.join('/app/output_local', f"{slot_anchor}_{llm_anch}.txt")
                else:
                    output_file = os.path.join('/app/output_relay', f"{slot_anchor}_{llm_anch}.txt")

                print(f"[RESUME] from {last_file}, index={last_index}")
                result = await handle_tar_file(tar_path_resume,
                                               output_file,
                                               model,
                                               processor,
                                               progress_csv_path,
                                               inference_start,
                                               start_index=last_index)
                if result == "DONE":
                    inf_finish = time.time()
                    await send_inference_result(sender_ip, local_ip, slot_anchor, llm_anch,
                                                inference_start, inf_finish, output_file)
                    await finalize_file(tar_path_resume, progress_csv_path, prompt_col, last_file)
                else:
                    # means "SKIPPED" => will not happen
                    if os.path.exists(tar_path_resume):
                        await asyncio.to_thread(os.remove, tar_path_resume)
                    await clear_progress(progress_csv_path)
                    print(f"[RESUME] Skipped tar => removed {tar_path_resume}")

        # 2) infinite loop
        while True:
            file_info = sorting(prompt_father_path, llm_anchor)
            if not file_info:
                print("[INFO] No new tar. Sleep 5s..")
                await asyncio.sleep(5)
                continue

            for slot_anchor, sender_ip, llm_anch, filename in file_info:
                # assigned check
                record = await prompt_col.find_one({"file_name": filename})
                if not record or not record.get("assigned"):
                    print(f"Tar {filename} not assigned => skip..")
                    continue

                await prompt_col.update_one(
                    {"file_name": filename},
                    {"$set": {"be_processed": True}}
                )

                # local or remote out path
                tar_path = os.path.join(prompt_father_path, filename)
                if sender_ip == local_ip:
                    output_file = os.path.join('/app/output_local', f"{slot_anchor}_{llm_anch}.txt")
                else:
                    output_file = os.path.join('/app/output_relay', f"{slot_anchor}_{llm_anch}.txt")

                # do handle
                inf_start = time.time()
                print(f"[INFO] Handling new tar {filename}, slot={slot_anchor}, anchor={llm_anch}")
                result = await handle_tar_file(tar_path,
                                               output_file,
                                               model,
                                               processor,
                                               progress_csv_path,
                                               inf_start,
                                               start_index=0)
                if result == "DONE":
                    inf_finish = time.time()
                    await send_inference_result(sender_ip, local_ip, slot_anchor, llm_anch,
                                                inf_start, inf_finish, output_file)
                    await finalize_file(tar_path, progress_csv_path, prompt_col, filename)
                else:
                    # "SKIPPED"
                    if os.path.exists(tar_path):
                        await asyncio.to_thread(os.remove, tar_path)
                    await clear_progress(progress_csv_path)
                    print(f"[SKIP] zero/empty tar => removed {tar_path}")

            print("[INFO] All tar tasks done. Wait more..")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
