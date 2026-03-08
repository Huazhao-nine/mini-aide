"""
底层 LLM 接口封装。

上层 `Agent` 并不关心具体用的是 DeepSeek、OpenAI 还是别的模型，只要求提供一个
`generate_response(messages)` 函数。这个文件负责把聊天消息发给后端模型，并把流式
返回拼接成最终文本。

从论文映射角度看，它不是 AIDE 的核心创新，而是 coding operator / review operator
背后的“模型调用适配层”：
- 论文里的 f(s, Σ(T)) 需要依赖 LLM 生成代码；
- 结构化评审也需要依赖 LLM 生成摘要或 JSON；
- 这个文件负责把这些上层需求转成具体 API 调用。

复试时可以把它概括为：论文方法本身在 `core/`，而 `backend/` 只是把方法接到真实大模型上。
"""

import sys
import os
from openai import OpenAI

# ==========================================
# 配置 DeepSeek API
# ==========================================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()
_CLIENT = None


def _get_client():
    """
    延迟初始化远程模型客户端。

    这里采用懒加载是纯工程取舍：避免模块导入时就触发网络请求或密钥错误。
    对论文方法本身没有改变，但能让 AIDE 在“先构造 Agent，再按需调用模型”时更稳。
    """
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

    从 AIDE 视角看，这个函数只负责“把模型输出拿回来”，不承担 search policy、
    summarization 或 evaluation 的业务判断。这样的边界划分有两个好处：
    1. 上层 `Agent` 可以保持与模型供应商解耦；
    2. coding operator 和 review operator 的策略都留在 `core/agent.py` 中统一维护。
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
