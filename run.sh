#!/bin/bash

# VITS TTS 一键运行脚本
# 安装依赖 → 启动 Flask 服务 → 打开浏览器

set -e

echo "=== VITS TTS 启动脚本 ==="

# 1. 安装依赖
echo "[1/3] 安装依赖..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt --break-system-packages
else
    pip install -r requirements.txt
fi

# 2. 启动 Flask 服务
echo "[2/3] 启动 Flask 服务..."
export FLASK_APP=server.py
python3 server.py &
SERVER_PID=$!

# 等待服务启动
sleep 3

# 3. 打开浏览器
echo "[3/3] 打开浏览器..."
if command -v open &> /dev/null; then
    open http://localhost:5000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000
fi

echo ""
echo "=== 服务已启动 ==="
echo "访问 http://localhost:5000 使用 TTS"
echo "按 Ctrl+C 停止服务"

# 等待用户中断
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM
wait $SERVER_PID
