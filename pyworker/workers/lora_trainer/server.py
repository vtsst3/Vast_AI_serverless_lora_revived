import os
import sys
# Vast.aiサーバーレス環境以外（ローカルテストなど）で実行する場合、
# REPORT_ADDR環境変数が存在しないため、ダミーの値を設定する。
if "REPORT_ADDR" not in os.environ:
    print("WARNING: 'REPORT_ADDR' not found in environment variables. Setting a dummy value.", file=sys.stderr)
    os.environ["REPORT_ADDR"] = "http://127.0.0.1:8080"

# レポートに基づき、WORKER_PORTが未設定の場合のフォールバックを追加
# Jupyterとのポート衝突を避けるため、デフォルトを8000に変更
if "WORKER_PORT" not in os.environ:
    print("WARNING: 'WORKER_PORT' not found. Defaulting to 8000 to avoid conflict with Jupyter.", file=sys.stderr)
    os.environ["WORKER_PORT"] = "8000"

import asyncio
import dataclasses
import glob
import logging
import os
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Type, Union

import boto3
import requests
import torch
from aiohttp import web, ClientResponse
from diffusers import DiffusionPipeline

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
try:
    from workers.lora_trainer.data_types import LoraJobPayload
except ImportError:
    from data_types import LoraJobPayload

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")
log = logging.getLogger(__name__)


# --- Global Settings & S3 Client ---
BASE_MODEL_PATH = os.environ.get('BASE_MODEL_PATH', '/workspace/modelse/checkpointse/Illustriouse/solventeclipseVpred_v11.safetensors')
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3_client = None

def initialize_s3_client():
    global s3_client
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not all([R2_ENDPOINT_URL, S3_BUCKET_NAME, access_key, secret_key]):
        log.warning("S3/R2 environment variables are not fully set. S3 operations will be skipped.")
        s3_client = None
        return
    s3_client = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    log.info("S3/R2 client initialized successfully.")


# --- Dummy Backend Server ---
# Since this worker is self-contained and doesn't proxy to another model server,
# we start a dummy HTTP server to satisfy PyWorker's health checks.

class DummyBackendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        return  # Suppress logging

def start_dummy_backend(port=1111):
    server = HTTPServer(('127.0.0.1', port), DummyBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Dummy backend server started on port {port}")

# --- Existing Training Functions (copied from the old server.py) ---
# These functions contain the core logic for the LoRA training process.

def run_command(command: list, cwd: str):
    log.info(f"Running command: {' '.join(command)} in {cwd}")
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in iter(process.stdout.readline, ''):
        log.info(line.strip())
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def run_tagger(train_data_dir: str) -> str:
    log.info("--- Starting Tagger ---")
    blacklist_path = '不要なタグを取り除くためのタグ一覧.txt'
    undesired_tags_str = ""
    try:
        with open(blacklist_path, 'r', encoding='utf-8') as f:
            tags_list = [line.strip() for line in f if line.strip()]
            undesired_tags_str = ",".join(tags_list)
            log.info(f"Loaded {len(tags_list)} tags from blacklist file: {blacklist_path}")
    except FileNotFoundError:
        log.warning(f"Blacklist file not found at '{os.path.abspath(blacklist_path)}'. Skipping undesired tags.")

    command = [
        "accelerate", "launch", "./sd-scripts/finetune/tag_images_by_wd14_tagger.py",
        train_data_dir, "--model_dir=/workspace/local_tagger_model", "--batch_size=1",
        "--caption_extension=.txt", "--general_threshold=0.35", "--character_threshold=0.85", "--onnx",
        "--remove_underscore",
    ]
    if undesired_tags_str:
        command.extend(["--undesired_tags", undesired_tags_str])

    run_command(command, cwd="/workspace/lora-worker")
    log.info("--- Tagger Finished ---")
    caption_files = glob.glob(os.path.join(train_data_dir, "*.txt"))
    return open(caption_files[0], 'r', encoding='utf-8').read() if caption_files else ""

def run_training(job_id: str, train_data_dir: str) -> str:
    output_dir = f"/workspace/output/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "accelerate", "launch", "./sd-scripts/sdxl_train_network.py",
        "--pretrained_model_name_or_path", BASE_MODEL_PATH, "--train_data_dir", train_data_dir,
        "--output_dir", output_dir, "--output_name", f"{job_id}_lora",
        "--resolution=1024,1024", "--train_batch_size=1", "--max_train_epochs=10",
        "--dataset_repeats=30", "--save_every_n_epochs=2", "--learning_rate=1.0",
        "--unet_lr=1.0", "--text_encoder_lr=1.0", "--network_module=networks.lora",
        "--network_dim=64", "--network_alpha=32", "--optimizer_type=Prodigy",
        '--optimizer_args', 'decouple=True', 'weight_decay=0.01', 'use_bias_correction=True', 'd_coef=0.8', 'd0=5e-5', 'safeguard_warmup=True', 'betas=0.9,0.99',
        "--lr_scheduler=cosine", "--mixed_precision=bf16", "--save_precision=bf16",
        "--gradient_checkpointing", "--xformers", "--no_half_vae", "--v_parameterization",
        "--min_snr_gamma=5", "--save_model_as=safetensors", "--caption_extension=.txt",
        "--cache_latents", "--cache_latents_to_disk"
    ]
    run_command(command, cwd="/workspace/lora-worker")
    log.info("--- LoRA Training Finished ---")
    lora_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    if not lora_files: raise FileNotFoundError("No LoRA file was generated.")
    return max(lora_files, key=os.path.getctime)

def generate_sample_image(lora_path: str, prompt: str, job_id: str) -> str:
    log.info("--- Loading pipeline for sample generation ---")
    pipeline = DiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, custom_pipeline="lpw_stable_diffusion_xl")
    pipeline.to("cuda")
    log.info(f"--- Loading LoRA weights from {lora_path} ---")
    pipeline.load_lora_weights(lora_path)
    log.info("--- Generating sample image ---")
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    sample_image_path = f"/workspace/sample_{job_id}.png"
    image.save(sample_image_path)
    log.info(f"--- Sample image saved to {sample_image_path} ---")
    return sample_image_path

def notify_backend(webhook_url: Optional[str], payload: Dict[str, Any]):
    if not webhook_url:
        log.info("Webhook URL not set. Skipping notification.")
        return
    try:
        log.info(f"Sending webhook to {webhook_url} with payload: {payload}")
        requests.post(webhook_url, json=payload, timeout=10)
    except requests.RequestException as e:
        log.error(f"Failed to send webhook: {e}")

def run_lora_training_process(payload: LoraJobPayload):
    job_id = payload.job_id
    try:
        log.info(f"--- Starting LoRA Training Job --- \nJobID: {job_id}")
        local_image_path = f"/workspace/{os.path.basename(payload.image_r2_key)}"
        
        if s3_client:
            log.info(f"Downloading {payload.image_r2_key} from R2...")
            s3_client.download_file(S3_BUCKET_NAME, payload.image_r2_key, local_image_path)
        else:
            raise RuntimeError("S3 client is not initialized. Cannot download image.")

        train_image_dir = f"/workspace/train_data/{job_id}/30_mychar"
        os.makedirs(train_image_dir, exist_ok=True)
        os.rename(local_image_path, os.path.join(train_image_dir, "image.jpg"))

        tagger_model_dir = "/workspace/local_tagger_model"
        if not os.path.exists(os.path.join(tagger_model_dir, "model.onnx")):
             raise FileNotFoundError("Tagger model not found. It should be downloaded by the on-start script.")

        caption = run_tagger(train_image_dir)
        log.info(f"Generated Caption: '{caption}'")

        lora_file_path = run_training(job_id, f"/workspace/train_data/{job_id}")

        lora_s3_key = f"users/{payload.user_id}/artifacts/{job_id}/lora.safetensors"
        if s3_client:
            log.info(f"Uploading LoRA to r2://{S3_BUCKET_NAME}/{lora_s3_key}")
            s3_client.upload_file(lora_file_path, S3_BUCKET_NAME, lora_s3_key)

        sample_prompt = f"masterpiece, best quality, 1girl, {caption}"
        sample_image_path = generate_sample_image(lora_file_path, sample_prompt, job_id)

        sample_image_s3_key = f"users/{payload.user_id}/artifacts/{job_id}/sample.png"
        if s3_client:
            log.info(f"Uploading sample image to r2://{S3_BUCKET_NAME}/{sample_image_s3_key}")
            s3_client.upload_file(sample_image_path, S3_BUCKET_NAME, sample_image_s3_key)

        completion_payload = {
            "job_id": job_id, "status": "COMPLETED",
            "artifacts": {"lora_url": f"r2://{S3_BUCKET_NAME}/{lora_s3_key}", "sample_image_url": f"r2://{S3_BUCKET_NAME}/{sample_image_s3_key}"}
        }
        notify_backend(payload.webhook_url, completion_payload)
        log.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        log.exception(f"An error occurred during LoRA training job {job_id}: {e}")
        error_payload = {"job_id": job_id, "status": "FAILED", "error": str(e)}
        notify_backend(payload.webhook_url, error_payload)


# --- PyWorker Framework Implementation ---

@dataclasses.dataclass
class LoraTrainHandler(EndpointHandler[LoraJobPayload]):
    @property
    def endpoint(self) -> str:
        return "/train-lora"

    @property
    def healthcheck_endpoint(self) -> str:
        # Since we don't use a separate model server, we can return a dummy path or None if allowed.
        # Returning a path that is likely to exist on the dummy model server URL.
        return "/"

    @classmethod
    def payload_cls(cls) -> Type[LoraJobPayload]:
        return LoraJobPayload

    def generate_payload_json(self, payload: LoraJobPayload) -> Dict[str, Any]:
        # This worker doesn't forward to another model server, so this is not used.
        return {}

    def make_benchmark_payload(self) -> LoraJobPayload:
        # Create a dummy payload for benchmarking.
        return LoraJobPayload.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        
        # This handler receives the job and starts it in the background.
        # It does not wait for the result but immediately returns an ACCEPTED response.
        try:
            auth_data = self.backend.get_auth_data(client_request)
            payload = await self.backend.get_payload(client_request, auth_data)
            
            log.info(f"Accepted job {payload.job_id}. Starting training process in background.")
            
            # Run the long-running training process in the background
            asyncio.create_task(run_lora_training_process(payload))
            
            return web.json_response({"status": "ACCEPTED", "job_id": payload.job_id}, status=202)

        except JsonDataException as e:
            log.error(f"Invalid JSON payload: {e}")
            return web.json_response({"status": "FAILED", "error": f"Invalid JSON payload: {e.errors}"}, status=400)
        except Exception as e:
            log.error(f"Error in handler: {e}")
            return web.json_response({"status": "FAILED", "error": str(e)}, status=500)

# We don't have a separate model server, so we use a dummy URL.
# The `log_actions` will tell the Vast.ai system when the worker is ready.
backend = Backend(
    model_server_url="http://127.0.0.1:1111", # Dummy URL pointing to our local dummy server
    model_log_file=os.environ.get("MODEL_LOG", "/workspace/lora-worker/worker.log"), # Log file to monitor
    allow_parallel_requests=True,
    benchmark_handler=LoraTrainHandler(benchmark_runs=1, benchmark_words=1), # A simple benchmark
    log_actions=[
        # When this log message appears in the MODEL_LOG file, the worker is considered "Ready"
        (LogAction.ModelLoaded, "======== LoRA Worker Server starting"),
        (LogAction.Info, "--- Starting LoRA Training Job ---"),
        (LogAction.ModelError, "CRITICAL ERROR"),
    ],
)

# Define the routes for the PyWorker server
routes = [
    web.post("/train-lora", backend.create_handler(LoraTrainHandler())),
]

if __name__ == "__main__":
    # Start the dummy backend server to satisfy health checks
    start_dummy_backend(port=1111)

    # Initialize S3 client before starting the server
    initialize_s3_client()
    
    # Signal that the worker is ready (matches LogAction.ModelLoaded)
    # We write to the log file explicitly to ensure Backend picks it up
    log_file_path = os.environ.get("MODEL_LOG", "/workspace/lora-worker/worker.log")
    try:
        with open(log_file_path, "a") as f:
            f.write("======== LoRA Worker Server starting\n")
    except Exception as e:
        log.warning(f"Could not write to log file {log_file_path}: {e}")

    # Also print to stdout for debugging
    print("======== LoRA Worker Server starting", flush=True)
    log.info("======== LoRA Worker Server starting")
    
    # Start the PyWorker server
    start_server(backend, routes)
