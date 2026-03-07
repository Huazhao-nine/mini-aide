import re

def extract_python_code(text):
    """
    从 LLM 回复中提取 Python 代码块。

    这是 coding operator 的一个很关键的小工具。论文把搜索对象定义为“代码空间中的解”，
    所以上层必须稳定地把模型回复还原成真正可执行的脚本。这里采用较宽松的提取策略：
    1. 优先寻找 ```python ... ```；
    2. 否则接受普通 ``` ... ```；
    3. 多个代码块时拼接，兼容模型分段输出；
    4. 如果完全没有 Markdown，但文本看起来像代码，则兜底返回原文。
    """
    # 先移除潜在的 `<think>` 包裹内容，避免推理模型把思维链混进代码区。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 正则匹配 Markdown 代码块。这里允许语言标识缺省，因为很多模型会省略 `python`。
    pattern = r"```(?:\s*python|\s*py)?\n(.*?)```"  
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code_blocks = [m.strip() for m in matches]
        # 多块拼接的原因是：有些模型会先给 import，再补主体函数，最后补 main。
        full_code = "\n\n".join(code_blocks)
        return full_code    
    else:
        # fallback：如果看起来明显是裸代码，就直接返回，尽量不让一次生成机会浪费掉。
        keywords = ["def ", "import ", "class ", "print("]
        if any(k in text for k in keywords):
            print("⚠️ Warning: No markdown detected, assuming raw text is code.")
            return text.strip()        
        return ""
