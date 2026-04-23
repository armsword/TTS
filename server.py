"""Flask Web API for TTS"""
import os
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, send_file, Response
from infer import TTSInferencer

app = Flask(__name__)

# 全局推理器实例
inferencer = None


def get_inferencer():
    """获取或创建推理器实例"""
    global inferencer
    if inferencer is None:
        inferencer = TTSInferencer()
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
    from flask import send_from_directory
    return send_from_directory("webui", "index.html")


def create_app():
    """创建 Flask 应用"""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
