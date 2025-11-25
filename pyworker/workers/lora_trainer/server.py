import os
import sys
import asyncio
import dataclasses
import glob
import logging
import subprocess
import json
from typing import Any, Dict, List, Optional

import boto3
import requests
import torch
from diffusers import DiffusionPipeline

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)


# --- Global Settings & S3 Client ---
BASE_MODEL_PATH = os.environ.get('BASE_MODEL_PATH', '/workspace/modelse/checkpointse/Illustriouse/solventeclipseVpred_v11.safetensors')
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')
s3_client = None

def initialize_s3_client():
    global s3_client
    access_key = os.environ.get('R2_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    if not all([R2_ENDPOINT_URL, R2_BUCKET_NAME, access_key, secret_key]):
        log.warning("S3/R2 environment variables are not fully set. S3 operations will be skipped.")
        s3_client = None
        return
    s3_client = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    log.info("S3/R2 client initialized successfully.")


# --- Training Functions ---

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
    # Taggerのモデルパスはonstart.shで準備されることを期待
    tagger_model_dir = "/workspace/wd14-tagger"
    if not os.path.exists(os.path.join(tagger_model_dir, "model.onnx")):
         raise FileNotFoundError(f"Tagger model not found in {tagger_model_dir}. It should be downloaded by the on-start script.")

    # ブラックリストファイルのパスは固定
    blacklist_path = '/workspace/lora-worker/不要なタグを取り除くためのタグ一覧.txt'
    undesired_tags_str = ""
    try:
        with open(blacklist_path, 'r', encoding='utf-8') as f:
            tags_list = [line.strip() for line in f if line.strip()]
            undesired_tags_str = ",".join(tags_list)
            log.info(f"Loaded {len(tags_list)} tags from blacklist file: {blacklist_path}")
    except FileNotFoundError:
        log.warning(f"Blacklist file not found at '{blacklist_path}'. Skipping undesired tags.")

    command = [
        "accelerate", "launch", "./sd-scripts/finetune/tag_images_by_wd14_tagger.py",
        train_data_dir, "--model_dir", tagger_model_dir, "--batch_size=1",
        "--caption_extension=.txt", "--general_threshold=0.35", "--character_threshold=0.85", "--onnx",
        "--remove_underscore",
    ]
    if undesired_tags_str:
        command.extend(["--undesired_tags", undesired_tags_str])

    run_command(command, cwd="/workspace/lora-worker")
    log.info("--- Tagger Finished ---")
    
    # 生成されたキャプションファイルは一つだけのはず
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
        "--dataset_repeats=30", "--save_every_n_epochs=2", "--learning_rate=1e-4", # Adjusted LR
        "--unet_lr=1e-4", "--text_encoder_lr=1e-5", "--network_module=networks.lora",
        "--network_dim=128", "--network_alpha=64", # Adjusted dim/alpha
        "--optimizer_type=AdamW8bit", # Changed to AdamW8bit for better stability
        "--lr_scheduler=cosine_with_restarts", "--lr_warmup_steps=20", # Adjusted scheduler
        "--mixed_precision=bf16", "--save_precision=bf16",
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
        log.warning("Webhook URL not set. Skipping notification.")
        return
    try:
        log.info(f"Sending webhook to {webhook_url} with payload: {payload}")
        requests.post(webhook_url, json=payload, timeout=30)
        log.info("Webhook sent successfully.")
    except requests.RequestException as e:
        log.error(f"Failed to send webhook: {e}")

def main():
    # 環境変数からジョブ情報を取得
    job_id = os.environ.get('JOB_ID')
    user_id = os.environ.get('USER_ID')
    character_id = os.environ.get('CHARACTER_ID')
    uploaded_image_keys_str = os.environ.get('UPLOADED_IMAGE_KEYS')
    webhook_url = os.environ.get('WEBHOOK_URL')

    # 必須の環境変数をチェック
    if not all([job_id, user_id, character_id, uploaded_image_keys_str, webhook_url]):
        missing_vars = [var for var in ['JOB_ID', 'USER_ID', 'CHARACTER_ID', 'UPLOADED_IMAGE_KEYS', 'WEBHOOK_URL'] if not os.environ.get(var)]
        error_message = f"Missing required environment variables: {', '.join(missing_vars)}"
        log.critical(error_message)
        # Webhookで失敗を通知しようと試みる
        if webhook_url and job_id:
            notify_backend(webhook_url, {"jobId": job_id, "status": "failed", "error": error_message})
        sys.exit(1)

    try:
        uploaded_image_keys = json.loads(uploaded_image_keys_str)
        # 簡単な学習なので、最初の画像キーのみ使用
        image_r2_key = uploaded_image_keys[0]
    except (json.JSONDecodeError, IndexError) as e:
        error_message = f"Failed to parse UPLOADED_IMAGE_KEYS or it's empty: {e}"
        log.critical(error_message)
        notify_backend(webhook_url, {"jobId": job_id, "status": "failed", "error": error_message})
        sys.exit(1)

    try:
        initialize_s3_client()
        log.info(f"--- Starting LoRA Training Job --- \nJobID: {job_id}")

        # 画像をR2からダウンロード
        local_image_dir = f"/workspace/train_data/{job_id}/30_mychar"
        os.makedirs(local_image_dir, exist_ok=True)
        local_image_path = os.path.join(local_image_dir, "image_01.png") # 拡張子をpngに
        
        if s3_client:
            log.info(f"Downloading {image_r2_key} from R2 to {local_image_path}...")
            s3_client.download_file(R2_BUCKET_NAME, image_r2_key, local_image_path)
            log.info("Download complete.")
        else:
            raise RuntimeError("S3 client is not initialized. Cannot download image.")

        # タガーを実行
        caption = run_tagger(local_image_dir)
        log.info(f"Generated Caption: '{caption}'")

        # 学習を実行
        lora_file_path = run_training(job_id, f"/workspace/train_data/{job_id}")

        # 学習済みLoRAファイルをR2にアップロード
        lora_s3_key = f"users/{user_id}/artifacts/{job_id}/lora.safetensors"
        if s3_client:
            log.info(f"Uploading LoRA to r2://{R2_BUCKET_NAME}/{lora_s3_key}")
            s3_client.upload_file(lora_file_path, R2_BUCKET_NAME, lora_s3_key)

        # サンプル画像を生成
        sample_prompt = f"masterpiece, best quality, 1girl, {caption}"
        sample_image_path = generate_sample_image(lora_file_path, sample_prompt, job_id)

        # サンプル画像をR2にアップロード
        sample_image_s3_key = f"users/{user_id}/artifacts/{job_id}/sample.png"
        if s3_client:
            log.info(f"Uploading sample image to r2://{R2_BUCKET_NAME}/{sample_image_s3_key}")
            s3_client.upload_file(sample_image_path, R2_BUCKET_NAME, sample_image_s3_key)

        # バックエンドに完了を通知
        completion_payload = {
            "jobId": job_id, "status": "completed",
            "artifacts": {"lora_url": lora_s3_key, "sample_image_url": sample_image_s3_key}
        }
        notify_backend(webhook_url, completion_payload)
        log.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        log.exception(f"An error occurred during LoRA training job {job_id}: {e}")
        error_payload = {"jobId": job_id, "status": "failed", "error": str(e)}
        notify_backend(webhook_url, error_payload)
        sys.exit(1)

if __name__ == "__main__":
    main()
