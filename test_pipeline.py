import sys
import os
import time

# 确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.llm import generate_response
from core.interpreter import Interpreter, ExecutionResult
from core.journal import Journal
from utils.utils import extract_python_code

def test_full_pipeline():
    print("🚀 [System] Starting Mini-AIDE Pipeline Test...")
    print("=" * 60)

    # 1. 初始化组件
    # ------------------------------------------------------------------
    interpreter = Interpreter() # 自动使用 config.py 里的 workspace
    journal = Journal()
    
    print("✅ Component Initialization Complete.")

    # 2. 模拟阶段一：DRAFT (起草代码)
    # ------------------------------------------------------------------
    task_description = "写一个Python脚本，计算前20个斐波那契数列的数字，并打印列表。最后输出一行 'Score: 1.0' 表示成功。"
    
    print(f"\n👉 [Step 1] DRAFT Stage: {task_description}")
    
    # A. 构造 Prompt (模拟 Agent 的工作)
    draft_messages = [
        {"role": "system", "content": "You are a Python expert. Write code to solve the user's problem. Wrap code in ```python ... ```."},
        {"role": "user", "content": task_description}
    ]

    # B. 调用大模型 (测试流式输出)
    print("\n🤖 [LLM] Generating Code (Draft)...")
    content, thought = generate_response(draft_messages, temperature=0.2)
    
    # C. 提取代码
    code = extract_python_code(content)
    print(f"\n📦 [Parser] Extracted Code Length: {len(code)} chars")

    # D. 运行代码
    print("\n🏃 [Interpreter] Executing Code...")
    result = interpreter.run(code, filename="fib_draft.py")
    print(f"   Success: {result.success}")
    print(f"   Output Snippet: {result.output.strip()[:50]}...")

    # E. 存入 Journal (作为根节点)
    print("\n📝 [Journal] saving Draft Node...")
    root_node = journal.add_node(
        parent=None,
        code=code,
        thought=thought,
        stage="draft",
        exec_result=result
    )

    # 3. 模拟阶段二：IMPROVE (基于上一步优化)
    # ------------------------------------------------------------------
    # 假设我们想让它把输出格式改一下，这测试了“树的生长”
    print(f"\n👉 [Step 2] IMPROVE Stage: Optimize the code.")

    # A. 从 Journal 获取上下文 (测试记忆回溯)
    history_context = journal.get_history_trace(root_node)
    
    improve_prompt = f"""
    {history_context}
    
    User Request: The previous code is correct, but I want you to calculate the sum of these numbers as well.
    Update the code. Still output 'Score: 1.0' at the end.
    """

    improve_messages = [
        {"role": "system", "content": "You are a Python expert. Improve the code based on history."},
        {"role": "user", "content": improve_prompt}
    ]

    # B. 调用大模型
    print("\n🤖 [LLM] Generating Code (Improve)...")
    content_2, thought_2 = generate_response(improve_messages, temperature=0.2)
    
    # C. 提取与运行
    code_2 = extract_python_code(content_2)
    print("\n🏃 [Interpreter] Executing Improved Code...")
    result_2 = interpreter.run(code_2, filename="fib_improve.py")

    # D. 存入 Journal (作为 Draft 的子节点！)
    print("\n📝 [Journal] Saving Improve Node (Child of Draft)...")
    child_node = journal.add_node(
        parent=root_node,  # <--- 关键：链接到父节点
        code=code_2,
        thought=thought_2,
        stage="improve",
        exec_result=result_2
    )

    # 4. 最终验证：打印树结构
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🌳 Final Solution Tree Structure (Check for parent-child link):")
    journal.print_tree()
    
    # 验证是否找到了最佳节点
    best = journal.get_best_node()
    if best:
        print(f"🏆 Best Node Found: ID={best.node_id} (Score={best.score})")
    else:
        print("❌ No successful node found.")

if __name__ == "__main__":
    test_full_pipeline()