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
from core.agent_prompt import AgentPrompt
from core.interpreter import Interpreter, ExecutionResult
from core.journal import Journal, Node
from utils.utils import extract_python_code
from config import WORKSPACE_DIR, num_drafts, debug_prob, max_debug_depth, timeout


# 简单的颜色类，用于美化控制台
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class Agent:
    def __init__(self, max_steps, data_preview=None):
        self.interpreter = Interpreter(workspace_dir=WORKSPACE_DIR, timeout=timeout)
        self.journal = Journal()
        self.max_steps = max_steps
        self.timeout = timeout
        self.data_preview = data_preview  # 由 run.py 传入的初始数据摘要

        self.search_cfg = {
            "num_drafts": num_drafts,
            "debug_prob": debug_prob,
            "max_debug_depth": max_debug_depth,
        }

        # 核心改进：初始化 Prompt 管理器
        self.prompt_manager = AgentPrompt(task_desc="")

    def solve(self, task_description: str):
        # 更新任务描述到管理器
        self.prompt_manager.task_desc = task_description

        print(
            f"{Colors.HEADER}🎯 [System] 任务开始: {task_description[:50]}...{Colors.ENDC}"
        )

        for step in range(self.max_steps):
            print(
                f"\n{Colors.BOLD}{'=' * 20} [Step {step + 1}/{self.max_steps}] Planning {'=' * 20}{Colors.ENDC}"
            )
            print(f"{Colors.CYAN}🌳 当前搜索树状态:{Colors.ENDC}")
            self.journal.print_tree()

            # 1. Search Policy (搜索策略)
            parent_node = self._search_policy()

            # 2. Action (根据策略生成 Prompt 并调用 LLM)
            if parent_node is None:
                print(f"{Colors.CYAN}👉 决策: DRAFT (起草新方案){Colors.ENDC}")
                new_node = self._draft()
                stage = "draft"
            elif parent_node.is_buggy:
                print(
                    f"{Colors.YELLOW}👉 决策: DEBUG (修复节点 {parent_node.node_id}){Colors.ENDC}"
                )
                new_node = self._debug(parent_node)
                stage = "debug"
            else:
                print(
                    f"{Colors.BLUE}👉 决策: IMPROVE (优化节点 {parent_node.node_id}, 当前分: {parent_node.score}){Colors.ENDC}"
                )
                new_node = self._improve(parent_node)
                stage = "improve"

            # 3. Execution (解释器运行)
            new_node.stage = stage # 补全 stage 信息
            if parent_node:
                new_node.parent = parent_node
                parent_node.children.append(new_node)
            filename = f"step{step}-[{stage}]-({new_node.node_id}).py"
            print(f"🏃 [Interpreter] 正在运行: {filename} ...")
            result = self.interpreter.run(new_node.code, filename=filename)

            self._print_exec_log(result)

            # 4. Review (结果审查)
            print(f"{Colors.HEADER}🤔 [Reviewer] 正在评估运行结果...{Colors.ENDC}")
            review_data = self._review_execution(new_node.code, result)

            # 5. Update Memory (更新节点信息)
            self._update_node_with_result(new_node, result, review_data)
            self.journal.add_node(new_node)

            # 同步 Review 数据
            last_node = self.journal.nodes[-1]
            last_node.score = review_data.get("score", 0.0)
            last_node.is_buggy = review_data.get("is_bug", True)
            last_node.analysis = review_data.get("summary", "")

            if last_node.is_buggy:
                print(
                    f"{Colors.RED}📊 评估结果: BUG ❌ | 概要: {last_node.analysis}{Colors.ENDC}"
                )
            else:
                print(
                    f"{Colors.GREEN}📊 评估结果: PASS ✅ | 分数: {last_node.score:.4f}{Colors.ENDC}"
                )
                print(f"   💡 评价: {last_node.analysis}")

            # 提前结束检查
            if last_node.score >= 0.999:
                break

        # 任务总结与最佳方案恢复
        best_node = self.journal.get_best_node()
        self._print_summary()

        if best_node:
            print(
                f"\n{Colors.BLUE}🔄 [系统] 恢复最佳方案 (ID: {best_node.node_id}) ...{Colors.ENDC}"
            )
            self.interpreter.run(best_node.code, filename="BEST_SOLUTION.py")
            return best_node.code
        return None

    # ==========================================
    # 🧠 Logic: Search Policy (采样逻辑)
    # ==========================================
    def _search_policy(self) -> Optional[Node]:
        draft_nodes = [n for n in self.journal.nodes if n.stage == "draft"]
        if len(draft_nodes) < self.search_cfg["num_drafts"]:
            return None

        buggy_leaves = [
            n for n in self.journal.nodes if n.is_buggy and len(n.children) == 0
        ]
        current_step = len(self.journal.nodes)
        progress = current_step / self.max_steps

        dynamic_debug_prob = self.search_cfg["debug_prob"] * (1 - progress)
        if buggy_leaves and random.random() < dynamic_debug_prob:
            return random.choice(buggy_leaves)

        successful_nodes = [n for n in self.journal.nodes if n.success]
        if not successful_nodes:
            return None

        candidates = [n for n in successful_nodes if n.score > 0.4]
        if not candidates:
            candidates = successful_nodes

        scores = np.array([n.score for n in candidates])
        T = 1.5 * (1 - progress) + 0.1
        scale_factor = 20.0
        exp_scores = np.exp((scores * scale_factor - np.max(scores * scale_factor)) / T)
        probs = exp_scores / np.sum(exp_scores)

        return np.random.choice(candidates, p=probs)

    # ==========================================
    # 🎨 Actions: 使用 AgentPrompt 生成内容
    # ==========================================

    def _draft(self) -> Node:
        # 获取历史轨迹摘要
        history = (
            self.journal.get_history_trace(None)
            if self.journal.nodes
            else "这是第一个节点。"
        )
        # 获取 Prompt 字典并格式化为字符串
        prompt_dict = self.prompt_manager.get_draft_prompt(history, self.data_preview)

        sys_msg = prompt_dict.pop("系统消息 (Introduction)")
        user_msg = self._dict_to_formatted_str(prompt_dict)

        self._log_prompt("DRAFT", user_msg)
        return self._query_llm(sys_msg, user_msg)

    def _improve(self, parent_node: Node) -> Node:
        # 获取日志摘要
        journal_summary = self.journal.get_history_trace(parent_node)
        prompt_dict = self.prompt_manager.get_improve_prompt(
            journal_summary, parent_node.code
        )

        sys_msg = prompt_dict.pop("系统消息 (Introduction)")
        user_msg = self._dict_to_formatted_str(prompt_dict)

        self._log_prompt("IMPROVE", f"Optimizing Node {parent_node.node_id}")
        return self._query_llm(sys_msg, user_msg)

    def _debug(self, parent_node: Node) -> Node:
        prompt_dict = self.prompt_manager.get_debug_prompt(
            parent_node.code, parent_node.error, self.data_preview
        )

        sys_msg = prompt_dict.pop("系统消息 (Introduction)")
        user_msg = self._dict_to_formatted_str(prompt_dict)

        self._log_prompt("DEBUG", f"Fixing Node {parent_node.node_id}")
        return self._query_llm(sys_msg, user_msg)

    def _review_execution(self, code, result) -> Dict[str, Any]:
        # 1. 解释器层面的硬错误快速返回
        if not result.success:
            return {
                "is_bug": True,
                "score": 0.0,
                "summary": f"执行崩溃: {result.error[:200]}",
            }

        # 2. 获取 Prompt (所有指令都在这里面了)
        prompt_dict = self.prompt_manager.get_review_prompt(code, result.output)
        sys_msg = prompt_dict.pop("系统消息 (Introduction)")
        user_msg = self._dict_to_formatted_str(prompt_dict)

        try:
            content, _ = generate_response(
                [{"role": "user", "content": f"{sys_msg}\n\n{user_msg}"}], 
                temperature=0
            )
            
            # 4. JSON 清洗与解析逻辑
            # 移除可能的 markdown 代码块标记
            json_str = content.replace("```json", "").replace("```", "").strip()
            
            # 寻找 JSON 对象的边界 (防御性编程)
            if "{" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            
            return json.loads(json_str)

        except Exception as e:
            print(f"{Colors.RED}Review 解析异常: {e} | Content: {content[:50]}...{Colors.ENDC}")
            return {
                "is_bug": True,
                "score": 0.0,
                "summary": f"Review 解析失败: {str(e)}",
            }
    # ==========================================
    # 🛠️ Helpers (辅助工具)
    # ==========================================

    def _query_llm(self, sys_msg, user_msg) -> Node:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # 1. 获取原始响应
        # content 包含：[自然语言大纲] + [```python 代码 ```]
        # reasoning 包含：DeepSeek R1 的隐式思维链 (如果你想存也可以存)
        full_content, reasoning = generate_response(messages)
        
        # 2. 解析代码
        code = extract_python_code(full_content)
        
        # 3. 解析大纲 (Thought/Plan)
        # 逻辑：取第一个代码块 ``` 之前的所有文本作为大纲
        if "```" in full_content:
            # split 之后的第一个元素就是代码块前面的文本
            thought = full_content.split("```")[0].strip()
        else:
            # 如果没有代码块（比如 LLM 拒绝生成），则整体都是思考
            thought = full_content.strip()
            
        # 4. (可选) 如果你想把 R1 的深度思考也记录下来，可以拼接到 thought 里
        # thought = f"[DeepSeek Reasoning]\n{reasoning}\n\n[Plan]\n{thought}"

        return Node(code=code, thought=thought, output="", success=False)

    def _dict_to_formatted_str(self, data: Any, indent=0) -> str:
        """将复杂的字典结构转换为清晰的 Prompt 文本"""
        res = []
        if isinstance(data, dict):
            for k, v in data.items():
                res.append(f"{'  ' * indent}### {k}")
                res.append(self._dict_to_formatted_str(v, indent + 1))
        elif isinstance(data, list):
            for item in data:
                res.append(f"{'  ' * indent}* {item}")
        else:
            res.append(f"{'  ' * indent}{data}")
        return "\n".join(res)

    def _update_node_with_result(self, node, result, review_data):
        node.output = result.output
        node.error = result.error
        node.execution_time = result.execution_time
        node.success = result.success and not review_data.get("is_bug", True)

    def _log_prompt(self, type_name, prompt_content):
        print(f"\n{Colors.CYAN}--- [Prompt: {type_name}] ---{Colors.ENDC}")
        print(textwrap.indent(prompt_content.strip(), "    "))
        print(f"{Colors.CYAN}----------------------------{Colors.ENDC}\n")

    def _print_exec_log(self, result):
        if result.success:
            print(
                f"{Colors.GREEN}✅ 运行成功 ({result.execution_time:.2f}s){Colors.ENDC}"
            )
            lines = result.output.strip().split("\n")
            print(
                "📄 输出摘要:\n"
                + ("\n".join(lines[-5:]) if len(lines) > 5 else result.output)
            )
        else:
            print(f"{Colors.RED}❌ 运行失败: {result.error[:200]}...{Colors.ENDC}")

    def _print_summary(self):
        print(
            f"\n{Colors.BOLD}=" * 50 + "\n📊 执行总结\n" + "=" * 50 + f"{Colors.ENDC}"
        )
        self.journal.print_tree()
        best = self.journal.get_best_node()
        if best:
            print(
                f"{Colors.GREEN}🏆 最佳分数: {best.score} (ID: {best.node_id}){Colors.ENDC}"
            )
