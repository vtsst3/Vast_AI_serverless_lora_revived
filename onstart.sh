#!/bin/bash

# エラーが発生したらスクリプトを終了する
set -e

echo "=== Starting On-start Script for LoRA Training Worker ==="

# 必要なツールをインストール
echo "Installing tools (git, aria2)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git aria2

# ソースコードをクローン
echo "Cloning worker repository..."
git clone https://github.com/vtsst3/Vast_AI_serverless_lora_revived.git /workspace/lora-worker
cd /workspace/lora-worker

# Pythonの依存関係をインストール
echo "Installing Python dependencies..."
# bitsandbytesはLinux環境では通常通りインストールできるはず
pip install -r requirements_lora.txt
pip install aiohttp boto3 "diffusers==0.27.2" "transformers==4.38.2" "accelerate==0.28.0" "huggingface-hub==0.22.2"

# モデルをダウンロード
echo "Downloading models..."

# ベースモデル
BASE_MODEL_DIR="/workspace/models/checkpoints"
mkdir -p "$BASE_MODEL_DIR"
export BASE_MODEL_PATH="$BASE_MODEL_DIR/solventeclipseVpred_v11.safetensors"
if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "Downloading base model..."
    aria2c -x 16 -s 16 -d "$BASE_MODEL_DIR" "https://huggingface.co/ckpt/solvent-eclipse/resolve/main/solventeclipseVpred_v11.safetensors"
else
    echo "Base model already exists."
fi

# Taggerモデル
TAGGER_MODEL_DIR="/workspace/wd14-tagger"
mkdir -p "$TAGGER_MODEL_DIR"
if [ ! -f "$TAGGER_MODEL_DIR/model.onnx" ]; then
    echo "Downloading WD14 Tagger model..."
    aria2c -x 16 -s 16 -d "$TAGGER_MODEL_DIR" "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx"
    aria2c -x 16 -s 16 -d "$TAGGER_MODEL_DIR" "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/selected_tags.csv"
else
    echo "Tagger model already exists."
fi

# メインの学習スクリプトを実行
echo "Starting LoRA training script..."
python /workspace/lora-worker/pyworker/workers/lora_trainer/server.py

echo "=== On-start Script Finished ==="
