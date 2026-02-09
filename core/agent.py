import json
import logging
import random
import time
import humanize
import textwrap
from typing import Optional, Dict, Any
import numpy as np
# 引入你的基础设施
from backend.llm import generate_response
from core.interpreter import Interpreter, ExecutionResult
from core.journal import Journal, Node
from utils.utils import extract_python_code
from config import WORKSPACE_DIR,num_drafts,debug_prob,max_debug_depth
# 简单的颜色类，用于美化控制台
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class Agent:
    def __init__(self, max_steps: int = 10, timeout: int = 6000):
        self.interpreter = Interpreter(workspace_dir=WORKSPACE_DIR, timeout=timeout)
        self.journal = Journal()
        self.max_steps = max_steps
        self.timeout = timeout
        
        self.search_cfg = {
            "num_drafts": num_drafts,
            "debug_prob": debug_prob,
            "max_debug_depth": max_debug_depth
        }

    def solve(self, task_description: str):
        print(f"{Colors.HEADER}🎯 [System] 任务开始: {task_description[:50]}...{Colors.ENDC}")

        for step in range(self.max_steps):
            print(f"\n{Colors.BOLD}{'='*20} [Step {step + 1}/{self.max_steps}] Planning {'='*20}{Colors.ENDC}")
            print(f"{Colors.CYAN}🌳 当前搜索树状态:{Colors.ENDC}")
            self.journal.print_tree()
            # 1. Search Policy
            parent_node = self._search_policy()
            if parent_node:
                print(f"👉 策略选择: 基于节点 {Colors.BOLD}{parent_node.node_id}{Colors.ENDC} (Score: {parent_node.score:.4f})")
            else:
                print("👉 策略选择: 无可用节点或探索新方向 -> 准备 DRAFT")
            # 2. Action
            if parent_node is None:
                print(f"{Colors.CYAN}👉 决策: DRAFT (起草新方案){Colors.ENDC}")
                new_node = self._draft(task_description)
                stage = "draft"
            elif parent_node.is_buggy:
                print(f"{Colors.YELLOW}👉 决策: DEBUG (修复节点 {parent_node.node_id}){Colors.ENDC}")
                new_node = self._debug(parent_node, task_description)
                stage = "debug"
            else:
                print(f"{Colors.BLUE}👉 决策: IMPROVE (优化节点 {parent_node.node_id}, 当前分: {parent_node.score}){Colors.ENDC}")
                new_node = self._improve(parent_node, task_description)
                stage = "improve"

            # 3. Execution
            filename = f"step_{step}_{stage}.py"
            print(f"🏃 [Interpreter] 正在运行: {filename} ...")
            result = self.interpreter.run(new_node.code, filename=filename)
            
            # 打印运行日志 (带截断)
            self._print_exec_log(result)

            # 4. Review
            print(f"{Colors.HEADER}🤔 [Reviewer] 正在评估运行结果...{Colors.ENDC}")
            review_data = self._review_execution(task_description, new_node.code, result)
            
            # 5. Update Memory
            self._update_node_with_result(new_node, result, review_data)
            self.journal.add_node(
                parent=parent_node,
                code=new_node.code,
                thought=new_node.thought,
                stage=stage,
                exec_result=result
            )
            
            # 手动同步 Review 数据
            last_node = self.journal.nodes[-1]
            last_node.score = review_data['score']
            last_node.is_buggy = review_data['is_bug']
            last_node.analysis = review_data['summary']
            
            # 打印最终判定
            if last_node.is_buggy:
                print(f"{Colors.RED}📊 评估结果: BUG ❌ (系统错误或结果无效){Colors.ENDC}")
            else:
                print(f"{Colors.GREEN}📊 评估结果: PASS ✅ | 分数: {last_node.score:.4f}{Colors.ENDC}")
                print(f"   💡 评价: {review_data['summary']}")

            # 6. Check Termination
            if last_node.score >= 0.99:
                print(f"\n{Colors.GREEN}🎉 找到满分解决方案！提前结束。{Colors.ENDC}")
                break

        best_node = self.journal.get_best_node()
        
        self._print_summary()

        if best_node:
            print(f"\n{Colors.BLUE}🔄 [系统] 正在恢复最佳方案 (ID: {best_node.node_id}) ...{Colors.ENDC}")
            print(f"   目标分数: {best_node.score}")
            
            # 重新运行最佳代码，以生成对应的 submission.csv
            # 我们用一个特殊的文件名 BEST_SOLUTION.py 来标识
            final_result = self.interpreter.run(best_node.code, filename="BEST_SOLUTION.py")
            
            if final_result.success:
                print(f"{Colors.GREEN}✅ 最佳方案已重现！submission.csv 已更新。{Colors.ENDC}")
                return best_node.code
            else:
                print(f"{Colors.RED}⚠️ 警告：最佳方案重运行时失败（可能是随机性导致）。{Colors.ENDC}")
                return best_node.code
        else:
            print(f"{Colors.RED}💀 未找到成功方案。{Colors.ENDC}")
            return None


    # ==========================================
    # 🧠 Search Policy (升级版：模拟退火 Softmax)
    # ==========================================
    def _search_policy(self) -> Optional[Node]:
        # 1. [Draft] 数量不足先起草 (最高优先级)
        # ------------------------------------------------
        draft_nodes = [n for n in self.journal.nodes if n.stage == "draft"]
        if len(draft_nodes) < self.search_cfg["num_drafts"]:
            return None

        # 2. [Debug] 随机给错误节点修复机会
        # ------------------------------------------------
        buggy_leaves = [
            n for n in self.journal.nodes   
            if n.is_buggy and len(n.children) == 0 
        ]
        # 动态 Debug 概率：步数越少越爱修 Bug，步数多了就不修了，专心冲刺
        current_step = len(self.journal.nodes)
        progress = current_step / self.max_steps
        
        # Debug 概率从配置值衰减到 0.1
        dynamic_debug_prob = self.search_cfg["debug_prob"] * (1 - progress * 0.8)
        
        if buggy_leaves and random.random() < dynamic_debug_prob:
            return random.choice(buggy_leaves)

        # 3. [Improve] 模拟退火 Softmax 采样
        # ------------------------------------------------
        # 筛选候选：成功的 + 未枯竭的(<3个孩子)
        candidates = [
            n for n in self.journal.nodes 
            if n.success and len(n.children) < 3
        ]

        if not candidates:
            # 兜底：如果没有符合条件的，退化为贪心，选全场最佳
            return self.journal.get_best_node()

        # --- 核心算法：带温度的 Softmax ---
        
        # A. 获取分数向量
        scores = np.array([n.score for n in candidates])
        
        # B. 计算动态温度 T (Simulated Annealing)
        # T 从 2.0 (高温, 接近均匀分布) 线性降低到 0.1 (低温, 接近 argmax)
        # progress: 0.0 -> 1.0
        T = 2.0 * (1 - progress) + 0.1
        
        # C. 计算 Softmax 概率
        # 公式: P(x) = exp(score / T) / sum(...)
        # 技巧: score 先减去最大值防止 exp 溢出，数值更稳定
        # 注意: 这里的 score 必须是正数且越大越好
        exp_scores = np.exp((scores - np.max(scores)) / T)
        probs = exp_scores / np.sum(exp_scores)
        
        # D. 根据概率采样
        selected_node = np.random.choice(candidates, p=probs)
        
        # (可选) 打印调试信息，让你在控制台看到 Agent 的"思考"
        best_idx = np.argmax(scores)
        print(f"🌡️ [退火] Step {current_step}/{self.max_steps} | Temp={T:.2f} | 最佳节点概率={probs[best_idx]:.2f}")
        
        return selected_node

    # ==========================================
    # 🎨 Prompt Engineering
    # ==========================================
    @property
    def _env_prompt(self):
        return (
            "运行环境说明：你可以使用常见的 Python 数据科学库，如 numpy, pandas, scikit-learn, "
            "torch, xgboost, lightgbm。数据文件位于 './input' 目录。"
            f"代码执行超时时间为 {humanize.naturaldelta(self.timeout)}。"
        )

    def _log_prompt(self, type_name, prompt_content):
        """美化 Prompt 输出，方便你观察"""
        print(f"\n{Colors.CYAN}--- [Prompt: {type_name}] ---{Colors.ENDC}")
        # 只打印最后 5 行 Prompt，避免刷屏，或者打印全部
        # 这里为了你观察，打印全部但加缩进
        print(textwrap.indent(prompt_content.strip(), '    '))
        print(f"{Colors.CYAN}----------------------------{Colors.ENDC}\n")

    def _draft(self, task_desc) -> Node:
        sys_msg = "你是一位 Kaggle Grandmaster。请根据任务描述编写初始 Python 解决方案。"
        user_msg = (
            f"任务描述：\n{task_desc}\n\n"
            f"{self._env_prompt}\n"
            "要求：单文件脚本，必须打印 'Score: <value>'。"
        )
        self._log_prompt("DRAFT", user_msg)
        return self._query_llm(sys_msg, user_msg)

    def _improve(self, parent_node, task_desc) -> Node:
        sys_msg = "你是一位 Kaggle Grandmaster。请优化之前的代码以获得更高分数。"
        user_msg = (
            f"任务：{task_desc}\n\n"
            f"上一步代码：\n```python\n{parent_node.code}\n```\n"
            f"上一步输出 (Score: {parent_node.score})：\n{parent_node.output}\n\n"
            "请提供具体的优化计划和完整代码。"
        )
        self._log_prompt("IMPROVE", f"Based on Node {parent_node.node_id} (Score: {parent_node.score})")
        return self._query_llm(sys_msg, user_msg)

    def _debug(self, parent_node, task_desc) -> Node:
        sys_msg = "你是一位 Python 调试专家。请修复代码中的 Bug。"
        user_msg = (
            f"任务：{task_desc}\n\n"
            f"错误日志：\n{parent_node.error}\n\n" 
            "请分析原因并提供修复后的代码。"
        )
        self._log_prompt("DEBUG", f"Fixing Node {parent_node.node_id}\nError: {parent_node.error}")
        return self._query_llm(sys_msg, user_msg)

    def _query_llm(self, sys_msg, user_msg) -> Node:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        # Generate 会打印自己的流式输出
        content, thought = generate_response(messages)
        code = extract_python_code(content)
        return Node(code=code, thought=thought, output="", success=False)

    # ==========================================
    # ⚖️ Reviewer
    # ==========================================
    def _review_execution(self, task, code, result) -> Dict[str, Any]:
        if not result.success:
             return {
                "is_bug": True,
                "score": 0.0,
                "summary": f"系统级错误: {result.error[:100]}..."
            }

        prompt = f"""
        你是一位 AI 裁判。评估代码运行结果，如果代码可以正常运行，is_bug=False，反之is_bug=True。
        Task: {task}...
        Output:{result.output}
        Output JSON: {{"is_bug": bool, "score": float, "summary": str}}
        """
        
        try:
            content, _ = generate_response([{"role": "user", "content": prompt}], temperature=0)
            json_str = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_str)
            return data
        except Exception as e:
            print(f"{Colors.RED}⚠️ Review 解析失败: {e}{Colors.ENDC}")
            return {"is_bug": True, "score": 0.0, "summary": "Reviewer JSON Error"}

    def _update_node_with_result(self, node, result, review_data):
        node.output = result.output
        node.error = result.error
        node.execution_time = result.execution_time
        node.success = result.success and not review_data['is_bug']

    def _print_exec_log(self, result):
        """美化打印运行日志，防止刷屏"""
        if result.success:
            print(f"{Colors.GREEN}✅ 运行成功 ({result.execution_time:.2f}s){Colors.ENDC}")
            # 只打印包含 Score 的行或者最后几行
            lines = result.output.strip().split('\n')
            if len(lines) > 10:
                print(f"📄 输出摘要 (共 {len(lines)} 行):")
                print("..." + "\n".join(lines[-5:]))
            else:
                print("📄 输出:\n" + result.output.strip())
        else:
            print(f"{Colors.RED}❌ 运行失败 ({result.execution_time:.2f}s){Colors.ENDC}")
            print(f"📄 错误信息:\n{result.error.strip()[:500]} ...")

    def _print_summary(self):
        print(f"\n{Colors.BOLD}" + "="*50)
        print("📊 执行总结 (Execution Summary)")
        print("="*50 + f"{Colors.ENDC}")
        self.journal.print_tree()
        best = self.journal.get_best_node()
        if best:
            print(f"{Colors.GREEN}🏆 最佳分数: {best.score} (Node: {best.node_id}){Colors.ENDC}")
        else:
            print(f"{Colors.RED}💀 未找到成功方案。{Colors.ENDC}")