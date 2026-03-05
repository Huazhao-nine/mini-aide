import json
import logging
import random
import re
import textwrap
from typing import Any, Dict, Optional, Callable

from backend.llm import generate_response
from core.interpreter import Interpreter, ExecutionResult
from core.journal import Journal, Node
from utils.utils import extract_python_code
from config import WORKSPACE_DIR, num_drafts, debug_prob, max_debug_depth, timeout

# --- 控制台日志（最小配置）---
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中尽量提取第一个 JSON 对象。"""
    if not text:
        return None

    s = text.strip()

    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


class Agent:
    """简化版 AIDE（但对齐原版 loop / 记忆闭环 / 树搜索语义）。"""

    def __init__(
        self,
        task_prompt: str,
        workdir: str,
        interpreter_factory: Optional[Callable[[str, int], Interpreter]] = None,
        journal: Optional[Journal] = None,
    ):
        self.task_prompt = task_prompt
        self.workdir = workdir
        self.num_drafts = int(num_drafts)
        self.debug_prob = float(debug_prob)
        self.max_debug_depth = int(max_debug_depth)
        self.timeout = int(timeout)

        self.journal = journal or Journal()
        self.interpreter = (
            interpreter_factory(workdir, self.timeout)
            if interpreter_factory
            else Interpreter(workdir=workdir, timeout=self.timeout)
        )

    # -------------------- prompts --------------------
    def _system_prompt(self) -> str:
        return (
            "你是一个在自动实验循环里工作的 ML 工程师 Agent。\n"
            "每次只做小步、可对比的改动；优先稳定 baseline，再逐步改进。\n"
            "当被要求“评审执行结果”时，必须输出严格 JSON（不要输出任何多余文本）。\n"
        )

    def _draft_prompt(self) -> str:
        return textwrap.dedent(
            f"""
            {self.task_prompt}

            你现在处于【DRAFT】阶段（从零写 baseline）。
            要求：
            - 生成一个完整、可直接运行的 Python 脚本
            - 方案要稳健、不要花里胡哨
            - 最后一行必须打印：`FINAL_MSE=<number>`
            - 如果有测试集，必须写出 `./working/submission.csv`
            """
        ).strip()

    def _improve_prompt(self, node: Node) -> str:
        mem = self.journal.build_memory(node)
        return textwrap.dedent(
            f"""
            {self.task_prompt}

            你现在处于【IMPROVE】阶段（在当前最优方案上做小步改进）。
            当前 best 分数（MSE）：{node.score}

            当前代码：
            ```python
            {node.code}
            ```

            记忆 / 历史经验（用于避免重复踩坑）：
            {mem}

            要求：
            - 只做一个“小改动”（原子级改进：一个想法/一个改动点）
            - 必须可运行
            - 最后一行必须打印：`FINAL_MSE=<number>`
            - 如果有测试集，必须写出 `./working/submission.csv`
            """
        ).strip()

    def _debug_prompt(self, node: Node) -> str:
        mem = self.journal.build_memory(node)
        return textwrap.dedent(
            f"""
            {self.task_prompt}

            你现在处于【DEBUG】阶段（修复崩溃/协议错误/输出无效等问题）。
            下面这份代码执行失败或输出不合规：

            代码：
            ```python
            {node.code}
            ```

            执行输出（stdout+stderr）：
            ```text
            {node.output}
            ```

            记忆 / 历史经验：
            {mem}

            要求：
            - 最小改动修复问题
            - 必须可运行
            - 最后一行必须打印：`FINAL_MSE=<number>`
            - 如果有测试集，必须写出 `./working/submission.csv`
            """
        ).strip()

    # -------------------- policy --------------------
    def search_policy(self) -> Dict[str, Any]:
        """对齐原版 AIDE：
        1) draft 到足够 root
        2) 以 debug_prob 概率 debug 一个 buggy leaf（且 debug 深度受限）
        3) 否则 greedy improve 当前 best
        """
        if len(self.journal.draft_nodes) < self.num_drafts:
            return {"action": "draft", "parent": None}

        buggy_leaf = self.journal.sample_buggy_leaf()
        if buggy_leaf is not None and buggy_leaf.debug_depth < self.max_debug_depth:
            if random.random() < self.debug_prob:
                return {"action": "debug", "parent": buggy_leaf}

        best = self.journal.get_best_node()
        if best is None:
            return {"action": "draft", "parent": None}
        return {"action": "improve", "parent": best}

    # -------------------- LLM interaction --------------------
    def _llm_call(self, user_prompt: str) -> str:
        """适配你当前的 DeepSeek wrapper: generate_response(messages, temperature) -> (content, reasoning)"""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        resp = generate_response(messages=messages)

        if isinstance(resp, tuple) and len(resp) >= 1:
            return (resp[0] or "").strip()
        if isinstance(resp, str):
            return resp.strip()
        return ""

    def _generate_code(self, action: str, parent: Optional[Node]) -> str:
        if action == "draft":
            prompt = self._draft_prompt()
        elif action == "improve":
            assert parent is not None
            prompt = self._improve_prompt(parent)
        elif action == "debug":
            assert parent is not None
            prompt = self._debug_prompt(parent)
        else:
            raise ValueError(f"未知 action: {action}")

        raw = self._llm_call(prompt)
        code = extract_python_code(raw)
        return code.strip() or raw.strip()

    # -------------------- execution + review --------------------
    def _execute(self, code: str) -> ExecutionResult:
        return self.interpreter.run(code)

    def _programmatic_metric_extract(self, text: str) -> Optional[float]:
        if not text:
            return None
        m = re.search(r"FINAL_MSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        if not m:
            return None
        try:
            v = float(m.group(1))
            if v != v or v < 0:
                return None
            return v
        except Exception:
            return None

    def _llm_review(self, task_desc: str, code: str, output: str) -> Optional[Dict[str, Any]]:
        """AIDE 核心：高信息量的结构化评审（JSON）→ 写入 node.analysis，供后续 memory 使用。"""
        review_prompt = textwrap.dedent(
            f"""
            你是一个 Kaggle 老手，正在评审一次代码执行结果。

            你必须只输出【严格 JSON】（不要任何多余文字、不要 Markdown），并且必须包含以下 key：
            - is_bug (boolean)：是否认为这次运行“有问题”（崩溃/指标缺失/协议不合规/明显泄漏等）
            - summary (string)：如果有问题，给出明确的修复建议；如果没问题，用 2~3 句话总结发现，并给下一步改进方向
            - metric (number or null)：如果能拿到有效的验证 MSE，就填数值，否则填 null
            - lower_is_better (boolean)：本任务 MSE 越小越好，填 true

            判断规则：
            - 如果运行崩溃、缺 FINAL_MSE、submission 路径/列错误、明显泄漏、验证协议不对，都算 is_bug=true
            - metric 应该是 5 折 OOF MSE（如果输出里有）
            - summary 要“可执行”：能指导下一轮 debug/improve

            【任务描述】：
            {task_desc}

            【代码】：
            {code}

            【执行输出】：
            {output}
            """
        ).strip()

        resp = self._llm_call(review_prompt)
        obj = _extract_first_json_obj(resp)
        if not isinstance(obj, dict):
            return None

        out: Dict[str, Any] = {}
        out["is_bug"] = bool(obj.get("is_bug", False))
        out["summary"] = str(obj.get("summary", "")).strip()
        metric = obj.get("metric", None)
        out["metric"] = float(metric) if isinstance(metric, (int, float)) else None
        out["lower_is_better"] = bool(obj.get("lower_is_better", True))
        return out

    def _review_execution(self, code: str, result: ExecutionResult) -> Dict[str, Any]:
        combined = (result.output or "").strip()
        metric = self._programmatic_metric_extract(combined)

        llm = self._llm_review(self.task_prompt, code, combined)
        if llm is not None:
            if llm.get("metric") is None and metric is not None:
                llm["metric"] = metric
            if (not result.success) or (llm.get("metric") is None):
                llm["is_bug"] = True
            return llm

        # fallback（无 LLM JSON 时）
        if not result.success:
            return {
                "is_bug": True,
                "summary": (result.error or "执行失败")[:400],
                "metric": None,
                "lower_is_better": True,
            }

        if metric is None:
            return {
                "is_bug": True,
                "summary": "没有找到 FINAL_MSE；请确保脚本最后一行打印 FINAL_MSE=<number>。",
                "metric": None,
                "lower_is_better": True,
            }

        return {
            "is_bug": False,
            "summary": f"验证 OOF MSE = {metric:.6f}。建议在保持协议正确的前提下做小步特征/模型改进。",
            "metric": metric,
            "lower_is_better": True,
        }

    # -------------------- state update --------------------
    def _update_node_with_result(self, node: Node, code: str, result: ExecutionResult, review: Dict[str, Any]) -> None:
        node.code = code
        node.success = bool(result.success)
        node.output = (result.output or "").strip()
        node.error = (result.error or "").strip()
        node.execution_time = float(result.execution_time or 0.0)

        node.analysis = str(review.get("summary", "")).strip()

        is_bug = bool(review.get("is_bug", False))
        metric = review.get("metric", None)

        # 对齐 AIDE：执行失败 / 无指标 → 直接视为 buggy
        if (not node.success) or (metric is None):
            node.is_buggy = True
            node.score = 9999.0
        else:
            node.is_buggy = is_bug
            node.score = float(metric)

    # -------------------- main loop --------------------
    def run(self, max_steps) -> None:
        for step in range(int(max_steps)):
            policy = self.search_policy()
            action: str = policy["action"]
            parent: Optional[Node] = policy["parent"]

            best_before = self.journal.get_best_node()
            best_str = f"{best_before.score:.6f}（{best_before.node_id}）" if best_before else "None"
            parent_str = parent.node_id if parent else "None"

            logger.info(f"\n===== 第 {step+1}/{int(max_steps)} 轮 =====")
            logger.info(f"策略选择：action={action} | parent={parent_str} | 当前best={best_str}")

            logger.info("LLM：开始生成代码...")
            code = self._generate_code(action, parent)
            logger.info(f"LLM：代码生成完成（长度={len(code)}）")

            node = self.journal.add_node(code=code, parent=parent, stage=action)
            logger.info(f"节点创建：id={node.node_id} | stage={node.stage}")

            logger.info("执行：运行 working/solution.py ...")
            result = self._execute(code)
            logger.info(f"执行结果：success={result.success} | 用时={result.execution_time:.2f}s | err={result.error or '-'}")

            if result.output:
                tail = "\n".join(result.output.splitlines()[-30:])
                logger.info("输出末尾（最后 30 行）：\n" + tail)

            logger.info("评审：解析 FINAL_MSE + 结构化评审（bug/summary/metric）...")
            review = self._review_execution(code, result)

            self._update_node_with_result(node, code, result, review)
            logger.info(f"本轮结果：node={node.node_id} | MSE={node.score:.6f} | buggy={node.is_buggy}")

            if node.analysis:
                logger.info("评审摘要：" + node.analysis[:600])

            best_now = self.journal.get_best_node()
            best_now_str = f"{best_now.score:.6f}（{best_now.node_id}）" if best_now else "None"
            logger.info(f"当前最优：{best_now_str}")

            # 每轮打印树结构
            self.journal.print_tree()

            # 保存 journal 快照
            self.journal.save(WORKSPACE_DIR)

        best = self.journal.get_best_node()
        if best is not None:
            logger.info(f"✅ 最终最优节点：{best.node_id} | MSE={best.score:.6f}")
        else:
            logger.info("⚠️ 最终没有找到有效（非 buggy）的节点。")

        logger.info("最终树结构：")
        self.journal.print_tree()