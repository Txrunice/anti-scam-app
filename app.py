import os
import json
import re
import tempfile
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# 加载本地 .env 文件 (方便你在本地测试，部署到服务器时这行会被自动忽略)
load_dotenv()

app = Flask(__name__)

# ==========================================
# ⚙️ 配置区域
# ==========================================

# 1. 获取 API Key
# 优先从系统环境变量获取，如果没有，则尝试读取 .env，最后报错
SILICON_API_KEY = os.environ.get("SILICON_API_KEY")

if not SILICON_API_KEY:
    # 为了防止你本地运行报错，可以在这里临时写死，但不要带着这行上传到 GitHub
    # SILICON_API_KEY = "sk-你的key写在这里"
    print("警告：未检测到 API Key，请设置环境变量 SILICON_API_KEY")

client = OpenAI(
    api_key=SILICON_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# 2. 模型选择 (已修复报错问题)
# 使用通义千问 2.5 (72B)，在硅基流动上非常稳定且免费/便宜
CHAT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
# 如果你想尝试 DeepSeek V3，解开下面这行（前提是你账号有权限）
# CHAT_MODEL = "deepseek-ai/DeepSeek-V3"

# 语音模型
AUDIO_MODEL = "FunAudioLLM/SenseVoiceSmall"

# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    temp_filepath = None
    try:
        text_content = ""

        # --- 步骤 1: 获取输入 (录音或文本) ---
        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            audio_file = request.files['audio_file']
            
            # 使用 tempfile 创建临时文件，适配 Linux/Render 服务器权限
            fd, temp_filepath = tempfile.mkstemp(suffix=".mp3")
            os.close(fd) # 关闭底层文件句柄
            
            audio_file.save(temp_filepath)
            
            # 调用 STT
            print(f"正在转写音频，使用模型: {AUDIO_MODEL}")
            with open(temp_filepath, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model=AUDIO_MODEL,
                    file=f
                )
            text_content = transcription.text
            
        elif 'text_input' in request.form and request.form['text_input'].strip() != '':
            text_content = request.form['text_input']
        else:
            return jsonify({"error": "请提供录音文件或输入文本"}), 400

        # --- 步骤 2: AI 分析 ---
        system_prompt = """
        你是一名反电信诈骗专家。分析用户提供的通话文本。
        请务必只返回纯 JSON 格式，不要包含 Markdown 标记（如 ```json）。
        JSON 字段要求：
        {
            "score": (0-100之间的整数，表示诈骗概率),
            "risk_level": ("低风险" | "中风险" | "极高风险"),
            "reasons": ["疑点1", "疑点2", "疑点3"],
            "advice": "给用户的简短建议"
        }
        """

        print(f"正在分析文本，使用模型: {CHAT_MODEL}")
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        
        # 清洗数据，防止 AI 返回 ```json 开头
        clean_content = re.sub(r"```json|```", "", raw_content).strip()
        
        analysis_result = json.loads(clean_content)
        analysis_result['transcript'] = text_content
        
        return jsonify(analysis_result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 清理临时文件
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)