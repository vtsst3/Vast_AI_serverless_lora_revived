#!/bin/bash

# エラーハンドリング: エラーが発生しても続行するように設定（デバッグ用）
# set -e を外しています

echo "=== Starting On-start Script ==="

# 既存のディレクトリを削除
rm -rf /workspace/lora-worker
rm -rf /workspace/vast_pyworker_lib

# 必要なツールとリポジトリを準備
echo "Installing git..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git

echo "Cloning repositories..."
git clone https://github.com/vtsst3/Vast_AI_serverless_lora_revived.git /workspace/lora-worker
git clone https://github.com/vast-ai/pyworker.git /workspace/vast_pyworker_lib

# 依存関係のインストール
echo "Installing dependencies..."
sed -i '/bitsandbytes/d' /workspace/lora-worker/requirements_lora.txt
pip install -r /workspace/lora-worker/requirements_lora.txt
pip install aiohttp boto3 "diffusers==0.27.2" "transformers==4.38.2" "accelerate==0.28.0" "huggingface-hub==0.22.2"

# PYTHONPATHを設定してライブラリを見つけられるようにする
export PYTHONPATH=/workspace/vast_pyworker_lib

# サーバーを起動し、ログをファイルにリダイレクト
echo "Starting server..."
cd /workspace/lora-worker
python pyworker/workers/lora_trainer/server.py > worker.log 2>&1 &

echo "Setup complete. Worker is running. Container will now idle."
