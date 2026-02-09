import re
import sys
from llama_cpp import Llama
from config import MODEL_PATH
_MODEL_INSTANCE = None
def get_model():
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
    提取 <think>...</think> 内部的思维链
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _clean_content(raw_text):
    """
    去除 <think> 部分，只返回最终的代码/回答
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    return cleaned.strip()

def generate_response(messages, temperature=0):
    """
    通用生成接口 (Streaming 版)
    :param messages: 标准 OpenAI 格式 [{"role": "user", "content": "..."}]
    :return: (content_str, thought_str)
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