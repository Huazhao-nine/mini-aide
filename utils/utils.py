import re

def extract_python_code(text):
    """
    从 LLM 的回复中提取 Python 代码块。
    策略：
    1. 优先寻找 ```python ... ``` 包裹的内容。
    2. 如果没有指定语言，寻找 ``` ... ``` 包裹的内容。
    3. 如果有多个代码块，默认将其拼接（适应分段输出的情况）。
    4. 如果完全没有 Markdown 标记，尝试直接返回文本（作为兜底，但会有风险）。
    """
    # 1. 移除 <think> 标签 (以防万一模型输出了思维链)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. 正则匹配：匹配 ```python 或 ``` 开头，直到下一个 ``` 结束
    # re.DOTALL 让 . 能匹配换行符
    # (?:python|py)? 表示可选的语言标识
    pattern = r"```(?:\s*python|\s*py)?\n(.*?)```"  
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # 找到了代码块
        # strip() 去除首尾空白
        code_blocks = [m.strip() for m in matches]
        # 将多个代码块用换行拼接（应对模型分段写代码的情况）
        full_code = "\n\n".join(code_blocks)
        return full_code    
    else:
        # 没找到代码块的 fallback 策略
        # 检查是否包含常见的 Python 关键字，如果包含，可能整个回复就是代码
        keywords = ["def ", "import ", "class ", "print("]
        if any(k in text for k in keywords):
            print("⚠️ Warning: No markdown detected, assuming raw text is code.")
            return text.strip()        
        return ""