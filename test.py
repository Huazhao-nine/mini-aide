
if __name__ == "__main__":
    # from utils.utils import extract_python_code
    # from core.interpreter import Interpreter
    # from backend.llm import generate_response
    # # 模拟测试
    # mock_messages = [
    #     {"role": "system", "content": "你是一个python代码高手，回答问题时，不要写if __name__ == __main__直接写main()，你只需要只回答不需要额外操作就可以直接运行的代码就可以，包含详细注释，代码应该用```python```的格式包裹起来。"},
    #     {"role": "user", "content": "写一个可以直接运行的python代码。"}
    # ]
    
    # # 这一步你应该能在控制台看到字一个个蹦出来，且包含 <think> 标签
    # final_content ,final_thought = generate_response(mock_messages)
    
    # print("\n✅ [Test Result] Parsing Check:")
    # print(f"Content Length: {len(final_content)} chars")
    # print(f"thought Length: {len(final_thought)} chars")
    # code = extract_python_code(final_content)
    # print("-" * 50)
    # print("截取py代码：\n"+code)
    # print("-" * 50)
    # interpreter = Interpreter()
    # res1 = interpreter.run(code)
    # print(f"Success: {res1.success}")
    # print(f"Output: {res1.output}")
    # print(f"Error Log: {res1.error}")
    from core.journal import Journal, ExecutionResult
        # 模拟数据
    j = Journal()
    
    # 1. Draft (Fail)
    res_fail = ExecutionResult(False, "running...", "SyntaxError: invalid syntax", 0.5)
    root = j.add_node(None, "print('hello')", "Thinking...", "draft", res_fail)
    
    # 2. Debug (Success, Score 0.6)
    res_ok_1 = ExecutionResult(True, "Score: 0.60\nRunning...", "", 1.2)
    node_fix = j.add_node(root, "print('hello world')", "Fix syntax", "debug", res_ok_1)
    
    # 3. Improve (Success, Score 0.8) - 基于 Node 2
    res_ok_2 = ExecutionResult(True, "Accuracy: 0.85\nDone.", "", 2.0)
    j.add_node(node_fix, "print('optimized')", "Optimize param", "improve", res_ok_2)

    # 4. Improve (Fail) - 另一条尝试分支
    res_fail_2 = ExecutionResult(False, "", "Timeout", 5.0)
    j.add_node(node_fix, "while True: pass", "Try heavy calc", "improve", res_fail_2)

    # 展示
    j.print_tree()
    print(f"Best Node Score: {j.get_best_node().score}")