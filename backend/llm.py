import sys
import os
from openai import OpenAI

# ==========================================
# 配置 DeepSeek API
# ==========================================
API_KEY = "sk-d66b659120d04f3eb60b79cd88b7b62a"
BASE_URL = "https://api.deepseek.com"
# export OPENAI_API_KEY="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"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_response(messages, temperature=0):
    """
    通用生成接口 (Streaming 版) - 适配 DeepSeek R1 API
    功能：仅负责流式接收并拼接字符串，不处理任何业务逻辑。
    返回：(full_content, full_reasoning)
    """
    print(f"🤖 [AI] Connecting to DeepSeek API (Temp: {temperature})...\n")
    print("-" * 50) 
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner", 
            messages=messages,
            stream=True
        )
        
        full_content = ""
        full_reasoning = ""
        is_thinking = True 
        
        sys.stdout.write("💭 [Thinking] \n")
        
        for chunk in response:
            # 1. 收集思维链 (Reasoning) - 仅 DeepSeek R1 有效
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content
                if reasoning:
                    # sys.stdout.write(reasoning)
                    # sys.stdout.flush()
                    full_reasoning += reasoning
            
            # 2. 收集正文 (Content)
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