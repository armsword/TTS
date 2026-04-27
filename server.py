"""Flask Web API for TTS"""
import os
import sys
import io

# MPS 不支持嵌套张量算子，需要 fallback 到 CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 添加 src 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from infer import TTSInferencer

app = Flask(__name__, static_folder='webui', static_url_path='')

# 全局推理器实例
inferencer = None


CHECKPOINT_DIR = "data/checkpoints_v2"


def get_latest_checkpoint():
    """获取最新的 checkpoint 路径

    优先使用 best_model.pt，否则使用最新的 epoch checkpoint
    """
    best = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best):
        return best

    import glob
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pt")))
    return checkpoints[-1] if checkpoints else None


def get_inferencer():
    """获取或创建推理器实例"""
    global inferencer
    if inferencer is None:
        checkpoint_path = get_latest_checkpoint()
        inferencer = TTSInferencer(checkpoint_path=checkpoint_path)
    else:
        # 每次请求时检查是否有新的 checkpoint
        new_checkpoint = get_latest_checkpoint()
        if new_checkpoint and (inferencer is not None and
                getattr(inferencer, '_checkpoint_path', None) != new_checkpoint):
            print(f"[Server] 检测到新 checkpoint: {new_checkpoint}，重新加载模型...")
            inferencer = TTSInferencer(checkpoint_path=new_checkpoint)
    return inferencer


@app.route("/api/health", methods=["GET"])
def health():
    """T-128: GET /api/health 返回 {"status": "ok", "model_loaded": true}"""
    return jsonify({
        "status": "ok",
        "model_loaded": inferencer is not None
    })


@app.route("/api/tts", methods=["POST"])
def tts():
    """T-130: POST /api/tts 请求 {"text": "hello", "language": "en"} 返回 audio/wav 响应"""
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text = data.get("text")
    language = data.get("language", "en")

    if not text:
        return jsonify({"error": "text field is required"}), 400

    try:
        inf = get_inferencer()
        waveform = inf.synthesize(text, language)

        # 转换为 wav 格式的二进制
        import scipy.io.wavfile as wavfile
        buffer = io.BytesIO()
        wavfile.write(buffer, 22050, waveform)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="output.wav"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """T-137: 返回 index.html"""
    return send_from_directory("webui", "index.html")


def create_app():
    """创建 Flask 应用"""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
