"""
底层 LLM 接口封装。

上层 `Agent` 并不关心具体用的是 DeepSeek、OpenAI 还是别的模型，只要求提供一个
`generate_response(messages)` 函数。这个文件负责把聊天消息发给后端模型，并把流式
返回拼接成最终文本。
"""

import sys
import os
from openai import OpenAI

# ==========================================
# 配置 DeepSeek API
# ==========================================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-d66b659120d04f3eb60b79cd88b7b62a").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()
_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        if not API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，请先在环境变量中配置。")
        # 懒加载 client，避免模块导入时立刻触发网络或配置错误。
        _CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _CLIENT

def generate_response(messages, temperature=0):
    """
    通用生成接口（流式版）。

    设计上刻意保持“只做 I/O，不做业务判断”：
    - 不解析 prompt；
    - 不提取代码；
    - 不做评审逻辑；
    这些都交由 `Agent` 处理。这里仅负责把模型输出稳定拿回来。

    返回 `(full_content, full_reasoning)`，便于上层在需要时丢弃思维链、只保留正文。
    """
    print(f"🤖 [AI] Connecting to DeepSeek API (Temp: {temperature})...\n")
    print("-" * 50) 
    
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )
        
        full_content = ""
        full_reasoning = ""
        is_thinking = True 
        
        sys.stdout.write("💭 [Thinking] \n")
        
        for chunk in response:
            # 1. 收集 reasoning_content：主要兼容 DeepSeek R1 这类会额外返回推理过程的模型。
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content
                if reasoning:
                    # sys.stdout.write(reasoning)
                    # sys.stdout.flush()
                    full_reasoning += reasoning
            
            # 2. 收集最终正文内容。上层真正会用于提取代码、JSON 评审的就是这部分。
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    # UI 交互：从思考切换到正文时打印分割线
                    if is_thinking and full_reasoning:
                        sys.stdout.write("\n\n💡 [Answer] \n")
                        is_thinking = False
                    elif is_thinking and not full_reasoning:
                         # 如果没有思维链直接输出内容（兼容 V3）
                        sys.stdout.write("\n💡 [Answer] \n")
                        is_thinking = False

                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_content += content

        print("\n" + "-" * 50 + "\n")
        
        return full_content, full_reasoning

    except Exception as e:
        print(f"\n❌ [API Error] {str(e)}")
        return "", ""
