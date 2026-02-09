import sys
import os
from openai import OpenAI

# ==========================================
# 1. 配置 DeepSeek API
# ==========================================
# 你的 API Key
API_KEY = "sk-d66b659120d04f3eb60b79cd88b7b62a"
BASE_URL = "https://api.deepseek.com"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 2. 核心生成函数 (流式)
# ==========================================

def generate_response(messages, temperature=0):
    """
    通用生成接口 (Streaming 版) - 适配 DeepSeek R1 API
    
    :param messages: 标准 OpenAI 格式 [{"role": "user", "content": "..."}]
    :return: (content_str, thought_str)
    """
    print(f"🤖 [AI] Connecting to DeepSeek API (Temp: {temperature})...\n")
    print("-" * 50) 
    
    try:
        # 使用 deepseek-reasoner 模型 (即 R1)
        response = client.chat.completions.create(
            model="deepseek-reasoner", 
            messages=messages,
            stream=True,
            temperature=temperature
        )
        
        full_content = ""
        full_reasoning = ""
        is_thinking = True # 标记当前是否在思考阶段

        # 打印思考开始标记
        sys.stdout.write("💭 [Thinking] \n")
        
        for chunk in response:
            # 1. 处理思维链 (Reasoning Content)
            # DeepSeek API 会先返回 reasoning_content，再返回 content
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content
                if reasoning:
                    sys.stdout.write(reasoning)
                    sys.stdout.flush()
                    full_reasoning += reasoning
            
            # 2. 处理正文 (Content)
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    # 如果是从思考转到正文，打印一个分割线
                    if is_thinking:
                        sys.stdout.write("\n\n💡 [Answer] \n")
                        is_thinking = False
                    
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_content += content

        print("\n" + "-" * 50 + "\n")
        
        # API 已经帮我们分好了，不需要再正则提取 <think> 标签了
        return full_content, full_reasoning

    except Exception as e:
        print(f"\n❌ [API Error] {str(e)}")
        # 返回空字符串防止程序崩溃
        return "", ""
