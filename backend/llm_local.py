"""
本地 LLM 推理适配层。

这个文件和 `backend/llm.py` 的职责相同，区别只是把远程 API 调用替换为本地模型推理。
它同样不属于论文中的核心算法创新，而是把论文里的 coding/review operator 接到
本地推理后端上的工程适配。

保留这份实现的意义主要有两点：
1. 方便在没有外部 API 或想离线演示时运行 mini-aide；
2. 说明本项目的核心逻辑不依赖某一家模型服务，而是依赖“给定消息，返回文本”的接口。
"""

import re
import sys
from llama_cpp import Llama
from utils.config import MODEL_PATH
_MODEL_INSTANCE = None


def get_model():
    """
    懒加载本地模型实例。

    这里把模型缓存为单例，是因为 AIDE 会在多轮搜索中频繁调用 LLM。若每轮都重新加载，
    会把大量时间浪费在模型初始化，而不是论文真正关注的“代码空间搜索”上。
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        print(f"🚀 [System] Loading Local Model from: 【{MODEL_PATH} 】")
        _MODEL_INSTANCE = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=8192,     
            verbose=False   
        )
        print("✅ [System] Model Loaded Successfully!")
    return _MODEL_INSTANCE

def _extract_thought(raw_text):
    """
    提取 `<think>...</think>` 内部的思维链。

    这是对推理模型输出格式的兼容性处理。论文并不要求显式保留思维链，但工程上常常需要把
    “最终可执行代码”和“模型中间推理文本”拆开，否则后续代码提取会被污染。
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _clean_content(raw_text):
    """
    去除 `<think>` 部分，只返回最终可执行正文。
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    return cleaned.strip()

def generate_response(messages, temperature=0):
    """
    本地模型的统一生成接口（流式版）。

    输入输出协议刻意对齐远程版 `backend/llm.py`，这样上层 `Agent` 无需区分当前到底是
    远程 API 还是本地模型。这种“后端可替换、上层不感知”的设计，是论文方法能够稳定复现
    的一个工程前提。
    """
    llm = get_model() 
    print(f"🤖 [AI] Generating response (Temperature: {temperature})...\n")
    print("-" * 50)   
    # 1. 开启流式生成 (stream=True)
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=8192, 
        temperature=temperature,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=True  
    )   
    full_response = ""
    # 2. 实时处理流块 (Chunks)
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            content_piece = delta['content']          
            # A. 实时打印到控制台
            sys.stdout.write(content_piece)
            sys.stdout.flush() # 强制刷新缓冲区，确保不卡顿            
            # B. 拼接到完整字符串中，用于后续正则解析
            full_response += content_piece
    print("\n" + "-" * 50 + "\n") 
    # 3. 生成完毕，进行后处理 (解析思维链和正文)
    thought = _extract_thought(full_response)
    content = _clean_content(full_response)    
    return content , thought
