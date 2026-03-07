"""
论文 AIDE 的主循环实现。

如果把论文 3.1 节的记号映射到这个文件，大致对应关系如下：
- solution tree T: `Journal` + `Node`
- search policy π(T): `search_policy()`
- summarization operator Σ(T): `Journal.generate_summary()` / `Journal.build_memory()`
- coding operator f(s, Σ(T)): `_generate_plan_and_code()`
- evaluator h(s): `_execute()` + `_review_execution()`

这个版本做了工程上的简化，但整体闭环仍然与论文一致：先生成候选代码，再运行评估，
把结果写回树结构，然后基于树中已有节点继续 draft / debug / improve。
"""

import json
import logging
import os
import random
import re
import shutil
import textwrap
from typing import Any, Dict, Optional, Callable, List, Tuple

from backend.llm import generate_response
from core.interpreter import Interpreter, ExecutionResult
from core.journal import Journal, Node
from core.metric import MetricValue, WorstMetricValue
from utils.utils import extract_python_code
from config import (
    WORKSPACE_DIR,
    num_drafts,
    debug_prob,
    max_debug_depth,
    timeout,
    top_k_candidates,
    explore_epsilon,
    draft_families,
)

# --- 控制台日志（最小配置）---
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 输出中尽量提取第一个 JSON 对象。

    这里服务于“结构化评审”环节：评审 prompt 要求模型只返回 JSON，但实际推理模型
    偶尔仍会包裹 Markdown 或夹杂额外文本，所以这里做一个鲁棒抽取。
    """
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
    """
    简化版 AIDE Agent。

    核心职责：
    1. 根据当前 solution tree 决定下一步是 draft / debug / improve；
    2. 调用 LLM 生成新的单文件 Python 方案；
    3. 运行方案并抽取 metric / 协议错误；
    4. 把结果追加到树中，形成论文里的 trial-and-error 搜索过程。
    """

    def __init__(
        self,
        task_prompt: str,
        workdir: str,
        interpreter_factory: Optional[Callable[[str, int], Interpreter]] = None,
        journal: Optional[Journal] = None,
    ):
        # 任务描述会被复用到所有 prompt 中，相当于论文里固定不变的 task context。
        self.task_prompt = task_prompt
        self.workdir = workdir
        self.num_drafts = int(num_drafts)
        self.debug_prob = float(debug_prob)
        self.max_debug_depth = int(max_debug_depth)
        self.timeout = int(timeout)
        self.top_k_candidates = max(1, int(top_k_candidates))
        self.explore_epsilon = max(0.0, min(1.0, float(explore_epsilon)))
        self.draft_families = [x.strip() for x in str(draft_families).split(",") if x.strip()]
        if not self.draft_families:
            self.draft_families = ["linear", "tree", "nn"]
        self.required_family = self._infer_required_family(self.task_prompt)
        if self.required_family and self.required_family not in self.draft_families:
            self.draft_families.insert(0, self.required_family)

        # `Journal` 保存整棵解空间树，`Interpreter` 充当无状态 evaluator h(s)。
        self.journal = journal or Journal()
        self.interpreter = (
            interpreter_factory(workdir, self.timeout)
            if interpreter_factory
            else Interpreter(workdir=workdir, timeout=self.timeout)
        )
        self.solution_dir = os.path.join(self.workdir, "solutions")
        os.makedirs(self.solution_dir, exist_ok=True)

    # -------------------- prompts --------------------
    def _infer_required_family(self, task_prompt: str) -> Optional[str]:
        # 这里是一个任务级硬约束解析器：如果 prompt 明确要求 NN 主线，后续 draft
        # 就优先从神经网络家族开始，避免搜索被树模型分支带偏。
        s = (task_prompt or "").lower()
        nn_keys = ["神经网络作业", "pytorch mlp", "nn 主线", "neural network homework"]
        if any(k in s for k in nn_keys):
            return "nn"
        return None

    def _system_prompt(self) -> str:
        # 所有 coding / review 调用共享的系统约束，确保 LLM 始终以“自动实验代理”身份工作。
        return (
            "你是一个在自动实验循环里工作的 ML 工程师 Agent。\n"
            "每次只做小步、可对比的改动；优先稳定 baseline，再逐步改进。\n"
            "当被要求“评审执行结果”时，必须输出严格 JSON（不要输出任何多余文本）。\n"
        )

    def _draft_prompt(self, family_hint: Optional[str] = None) -> str:
        """
        Drafting prompt，对应论文 3.2 节的 drafting 入口。

        在 AIDE 中，draft 的目标不是一次性得到最终最优解，而是先构造若干“可运行的起点”
        作为树的根节点，为后续 improve 提供可继续优化的 parent。
        """
        family_line = ""
        if family_hint:
            family_line = f"- 本轮 baseline 家族优先：`{family_hint}`（保持可运行和评估协议正确）\n"
        return textwrap.dedent(
            f"""
            {self.task_prompt}

            你现在处于【DRAFT】阶段（从零写 baseline）。
            要求：
            {family_line}- 生成一个完整、可直接运行的 Python 脚本
            - 方案要稳健、不要花里胡哨
            - 最后一行必须打印：`FINAL_SCORE=<number>`
            - 如果有测试集，必须写出 `./submission.csv`（并可额外同步到 `./working/submission.csv`）
            - 若使用 early stopping 保存最佳模型，必须使用 `copy.deepcopy(model.state_dict())` 保存并在预测前恢复
            """
        ).strip()

    def _improve_prompt(self, node: Node) -> str:
        """
        Improve prompt，对应论文里的 atomic improvement。

        这里会把父节点代码和历史摘要一起交给 LLM，但明确要求“只做一个小改动”，
        这样每个子节点都能被解释成对父节点的一次局部增量修改，便于树搜索比较。
        """
        mem = self.journal.build_memory(node)
        return textwrap.dedent(
            f"""
            {self.task_prompt}

            你现在处于【IMPROVE】阶段（在当前最优方案上做小步改进）。
            当前 best 分数（score）：{node.score}

            当前代码：
            ```python
            {node.code}
            ```

            记忆 / 历史经验（用于避免重复踩坑）：
            {mem}

            要求：
            - 只做一个“小改动”（原子级改进：一个想法/一个改动点）
            - 必须可运行
            - 最后一行必须打印：`FINAL_SCORE=<number>`
            - 如果有测试集，必须写出 `./submission.csv`（并可额外同步到 `./working/submission.csv`）
            - 若使用 early stopping 保存最佳模型，必须使用 `copy.deepcopy(model.state_dict())` 保存并在预测前恢复
            """
        ).strip()

    def _debug_prompt(self, node: Node) -> str:
        """
        Debug prompt，对应论文 3.2 节的 debugging 入口。

        输入不仅包含旧代码，还包含执行日志和局部历史，使模型能围绕“最小修复”而不是
        “完全重写”展开调试。
        """
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
            - 最后一行必须打印：`FINAL_SCORE=<number>`
            - 如果有测试集，必须写出 `./submission.csv`（并可额外同步到 `./working/submission.csv`）
            - 若使用 early stopping 保存最佳模型，必须使用 `copy.deepcopy(model.state_dict())` 保存并在预测前恢复
            """
        ).strip()

    # -------------------- policy --------------------
    def _choose_draft_family(self) -> str:
        # 论文里 drafting 阶段强调先拿到一组不同初始解。这里通过统计各 family 的草稿数，
        # 尽量让根节点覆盖不同模型家族，而不是一直重复同一种 baseline。
        if self.required_family:
            return self.required_family

        family_count = {f: 0 for f in self.draft_families}
        for n in self.journal.draft_nodes:
            if n.family_hint in family_count:
                family_count[n.family_hint] += 1

        min_count = min(family_count.values())
        candidates = [f for f, c in family_count.items() if c == min_count]
        return random.choice(candidates)

    def _topk_good_nodes(self) -> List[Node]:
        # improve 阶段不会在整棵树上盲选，而是先截出 top-k 候选，再做 exploitation /
        # exploration 权衡。这是对论文中“通常围绕当前最好节点继续优化”的一个温和扩展。
        good = [n for n in self.journal.nodes if n.success and not n.is_buggy]
        if self.required_family:
            preferred = [n for n in good if n.family_hint == self.required_family]
            if preferred:
                good = preferred
        good = sorted(good, key=lambda n: n.metric, reverse=True)
        return good[: self.top_k_candidates]

    def _pick_parent_from_topk(self, topk: List[Node]) -> Tuple[Node, str]:
        if len(topk) == 1:
            return topk[0], "best_only"

        if random.random() < self.explore_epsilon:
            # explore: 偏向“子节点更少”的分支，提升树搜索多样性
            weights = [1.0 / (1.0 + len(n.children)) for n in topk]
            parent = random.choices(topk, weights=weights, k=1)[0]
            return parent, "explore"

        # exploit: 优先更优指标，其次优先分支较浅（children 更少）
        best_metric = max(n.metric for n in topk)
        same_best = [n for n in topk if n.metric == best_metric]
        parent = min(same_best, key=lambda n: len(n.children))
        return parent, "greedy"

    def search_policy(self) -> Dict[str, Any]:
        """
        对齐论文中 search policy π(T) 的硬编码决策逻辑：
        1) draft 到足够 root
        2) 以 debug_prob 概率 debug 一个 buggy leaf（且 debug 深度受限）
        3) 否则在 top-k 中选择 improve parent（兼顾 exploitation / exploration）

        这正是论文 3.2 节对 search policy 的具体化：先有足够初始解，再优先处理可修复
        的错误分支，否则持续沿高价值分支做改进。
        """
        if len(self.journal.draft_nodes) < self.num_drafts:
            return {"action": "draft", "parent": None, "family_hint": self._choose_draft_family()}

        buggy_leaf = self.journal.sample_buggy_leaf()
        if buggy_leaf is not None and buggy_leaf.debug_depth <= self.max_debug_depth:
            if random.random() < self.debug_prob:
                return {"action": "debug", "parent": buggy_leaf}

        topk = self._topk_good_nodes()
        if not topk:
            return {"action": "draft", "parent": None, "family_hint": self._choose_draft_family()}

        parent, picked_by = self._pick_parent_from_topk(topk)
        return {"action": "improve", "parent": parent, "topk": topk, "picked_by": picked_by}

    # -------------------- LLM interaction --------------------
    def _llm_call(self, user_prompt: str) -> str:
        """
        对底层 LLM 接口做一层薄封装。

        这样上层不关心供应商返回的是纯字符串还是 `(content, reasoning)` 二元组，只消费
        最终正文即可。
        """
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

    def _extract_text_up_to_code(self, text: str) -> str:
        # 把代码块前面的 1-3 句解释提取出来，作为节点的 `thought`。
        # 这相当于给树中的每个节点附一份“改动意图”摘要，方便后续 Σ(T) 复用。
        if not text:
            return ""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r"```(?:\s*python|\s*py)?\n", clean, flags=re.IGNORECASE)
        if not m:
            return ""
        prefix = clean[: m.start()].strip()
        return prefix[:1000]

    def _generate_plan_and_code(
        self,
        action: str,
        parent: Optional[Node],
        family_hint: Optional[str] = None,
        retries: int = 3,
    ) -> Tuple[str, str]:
        """
        coding operator f(s, Σ(T)) 的主入口。

        不同 action 使用不同 prompt 模板，但统一产出 `(thought, code)`：
        - thought: 记录这次修改的高层意图；
        - code: 可执行脚本本体，也就是 solution tree 中的节点内容。
        """
        if action == "draft":
            prompt = self._draft_prompt(family_hint=family_hint)
        elif action == "improve":
            assert parent is not None
            prompt = self._improve_prompt(parent)
        elif action == "debug":
            assert parent is not None
            prompt = self._debug_prompt(parent)
        else:
            raise ValueError(f"未知 action: {action}")

        last_raw = ""
        for i in range(int(retries)):
            retry_tip = ""
            if i > 0:
                # 如果第一次没拿到代码，则显式提醒模型遵守“解释 + 单个代码块”的输出格式。
                retry_tip = (
                    "\n\n上一次回复未提取到可运行代码。"
                    "请严格按要求：先 1-3 句简短说明，再输出一个 ```python``` 代码块。"
                )
            raw = self._llm_call(prompt + retry_tip)
            last_raw = raw or ""
            code = extract_python_code(last_raw).strip()
            thought = self._extract_text_up_to_code(last_raw)
            if code:
                return thought, code

            logger.warning(f"LLM 代码提取失败，重试 {i + 1}/{int(retries)}")

        # fail-fast：避免空脚本被当成“执行成功但无指标”的噪声节点，污染树结构。
        fallback_code = "raise RuntimeError('LLM returned empty/invalid code after retries')\n"
        fallback_thought = "LLM generation failed after retries."
        return fallback_thought, fallback_code

    # -------------------- execution + review --------------------
    def _cleanup_submission_files(self) -> None:
        # 每轮执行前清理旧 submission，避免把上一轮遗留文件误判为当前节点的有效输出。
        stale_paths = [
            os.path.join(self.workdir, "submission.csv"),
            os.path.join(self.workdir, "working", "submission.csv"),
        ]
        for p in stale_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                continue

    def _execute(self, code: str) -> ExecutionResult:
        # evaluator h(s) 的第一步：真正运行候选脚本，拿到 stdout/stderr/异常/耗时。
        self._cleanup_submission_files()
        return self.interpreter.run(code)

    def _dump_solution_code(self, step_index: int, stage_name: str, node_id: str, code: str) -> str:
        # 为每个节点落盘快照，便于回溯 solution tree 的演化路径。
        filename = f"step{step_index}-{stage_name}-{node_id}.py"
        path = os.path.join(self.solution_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"代码快照保存失败: {path} | err={e}")
        return path

    def _save_best_code(self, best: Node) -> str:
        # 搜索结束后把当前全局最优节点单独保存，方便答辩时直接展示“最终版本”。
        path = os.path.join(self.workdir, "best.py")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(best.code)
        except Exception as e:
            logger.warning(f"best.py 保存失败: {path} | err={e}")
        return path

    def _promote_submission_file(self) -> Optional[str]:
        # 某些生成脚本会把提交文件写到 `working/submission.csv`，这里统一提升到 canonical 路径，
        # 方便最终检查和人工查看。
        nested = os.path.join(self.workdir, "working", "submission.csv")
        canonical = os.path.join(self.workdir, "submission.csv")

        if os.path.exists(nested):
            try:
                shutil.copy2(nested, canonical)
                return canonical
            except Exception as e:
                logger.warning(f"同步最终 submission 失败: {nested} -> {canonical} | err={e}")
                return None

        if os.path.exists(canonical):
            return canonical
        return None

    def _programmatic_metric_extract(self, text: str) -> Optional[float]:
        # 论文里 h(s) 需要是一个标量目标值。这里通过约定的 `FINAL_SCORE=...` 从输出中提取。
        if not text:
            return None
        m = re.search(r"FINAL_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        if not m:
            return None
        try:
            v = float(m.group(1))
            if v != v or v < 0:
                return None
            return v
        except Exception:
            return None

    def _is_final_score_last_line(self, text: str) -> bool:
        # 强制 FINAL_SCORE 出现在最后一行，是为了让指标抽取稳定、减少被中间日志污染的概率。
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if not lines:
            return False
        return re.match(r"^FINAL_SCORE\s*=\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", lines[-1]) is not None

    def _check_submission_file(self) -> list[str]:
        # 这属于“协议正确性”检查，不直接反映模型效果，但决定该节点是否可作为有效 solution。
        candidates = [
            os.path.join(self.workdir, "submission.csv"),
            os.path.join(self.workdir, "working", "submission.csv"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return ["缺少提交文件：submission.csv 或 working/submission.csv"]

        try:
            import pandas as pd
            sub = pd.read_csv(path, nrows=2)
        except Exception as e:
            return [f"提交文件读取失败：{e}"]

        required = self._infer_submission_columns()
        missing = [c for c in required if c not in sub.columns]
        if missing:
            return [f"提交文件缺少列：{missing}（当前列：{list(sub.columns)}）"]
        return []

    def _infer_submission_columns(self) -> list[str]:
        # 当前项目已有多个任务入口，submission 列名不能继续写死成 COVID 回归格式。
        prompt = self.task_prompt or ""
        checks = [
            (["PassengerId", "Survived"], ["PassengerId", "Survived"]),
            (["ImageId", "Label"], ["ImageId", "Label"]),
            (["id", "tested_positive_day3"], ["id", "tested_positive_day3"]),
        ]
        prompt_lower = prompt.lower()
        for needles, columns in checks:
            if all((f"`{x}`".lower() in prompt_lower) or (x.lower() in prompt_lower) for x in needles):
                return columns
        return ["id", "tested_positive_day3"]

    def _static_code_checks(self, code: str) -> list[str]:
        # 纯静态规则检查，主要抓容易在 Kaggle/训练脚本里引发 silent bug 的模式。
        errs: list[str] = []
        code_s = code or ""

        if "state_dict().copy()" in code_s and "copy.deepcopy(" not in code_s:
            errs.append(
                "检测到 `state_dict().copy()`：这是浅拷贝，可能导致最佳权重失效。"
                "请使用 `copy.deepcopy(model.state_dict())`。"
            )
        return errs

    def _programmatic_review(self, code: str, result: ExecutionResult) -> Dict[str, Any]:
        """
        程序化评审。

        这里尽量把“是否有效节点”的判定做成确定性逻辑，减少完全依赖 LLM 主观判断：
        - 是否运行成功；
        - 是否产出 FINAL_SCORE；
        - FINAL_SCORE 是否在最后一行；
        - submission 文件是否存在且列名正确；
        - 是否触发若干静态禁用模式。
        """
        combined = (result.output or "").strip()
        hard_errors: list[str] = []

        if not result.success:
            hard_errors.append(result.error or "执行失败")

        metric = self._programmatic_metric_extract(combined)
        if metric is None:
            hard_errors.append("没有解析到 FINAL_SCORE=<number>")
        elif not self._is_final_score_last_line(combined):
            hard_errors.append("FINAL_SCORE 不是最后一个非空输出行")

        if "submission.csv" in self.task_prompt:
            hard_errors.extend(self._check_submission_file())
        hard_errors.extend(self._static_code_checks(code))

        return {
            "metric": metric,
            "hard_errors": hard_errors,
            "is_compliant": len(hard_errors) == 0,
        }

    def _llm_review(self, task_desc: str, code: str, output: str) -> Optional[Dict[str, Any]]:
        """
        LLM 评审。

        论文里的 Σ(T) 不只是保存标量分数，还会保存“这次改动做了什么、哪里出了问题、
        下一步该往哪改”的摘要。这里让模型输出结构化 JSON，再写入 `node.analysis`，
        供后续 improve / debug prompt 使用。
        """
        review_prompt = textwrap.dedent(
            f"""
            你是一个 Kaggle 老手，正在评审一次代码执行结果。

            你必须只输出【严格 JSON】（不要任何多余文字、不要 Markdown），并且必须包含以下 key：
            - is_bug (boolean)：是否认为这次运行“有问题”（崩溃/指标缺失/协议不合规/明显泄漏等）
            - summary (string)：如果有问题，给出明确的修复建议；如果没问题，用 2~3 句话总结发现，并给下一步改进方向
            - metric (number or null)：如果能拿到有效的验证 score，就填数值，否则填 null
            - lower_is_better (boolean)：根据任务描述判断 score 是否越小越好

            判断规则：
            - 如果运行崩溃、缺 FINAL_SCORE、submission 路径/列错误、明显泄漏、验证协议不对，都算 is_bug=true
            - metric 应该与任务描述中的最终 score 定义一致（如果输出里有）
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
        # 最终评审以程序化检查为主、LLM 摘要为辅。前者负责“能不能进树”，后者负责“记住什么”。
        combined = (result.output or "").strip()
        prog = self._programmatic_review(code, result)
        llm = self._llm_review(self.task_prompt, code, combined)
        lower_is_better = True
        if isinstance(llm, dict) and isinstance(llm.get("lower_is_better"), bool):
            lower_is_better = bool(llm["lower_is_better"])

        summary_parts = []
        if prog["hard_errors"]:
            summary_parts.append("程序化校验失败：" + "；".join(prog["hard_errors"]))
        elif prog["metric"] is not None:
            summary_parts.append(f"程序化校验通过，FINAL_SCORE={float(prog['metric']):.6f}。")
        else:
            summary_parts.append("程序化校验未通过。")

        if llm is not None and llm.get("summary"):
            summary_parts.append(str(llm["summary"]).strip())

        return {
            "is_bug": not bool(prog["is_compliant"]),
            "summary": " ".join(summary_parts).strip(),
            "metric": prog["metric"],
            "lower_is_better": lower_is_better,
            "is_compliant": bool(prog["is_compliant"]),
            "hard_errors": list(prog["hard_errors"]),
        }

    # -------------------- state update --------------------
    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Dict[str, Any]:
        """
        把 evaluator 的结果写回 solution tree 节点。

        这是 Algorithm 1 中：
        - `vn = h(sn)` 得到分数；
        - `Tn = Tn-1 ∪ {node, edge}` 更新树；
        之间的桥接步骤。
        """
        node.absorb_exec_result(exec_result)
        review = self._review_execution(node.code, exec_result)

        node.analysis = str(review.get("summary", "")).strip()
        node.is_compliant = bool(review.get("is_compliant", False))
        node.compliance_errors = list(review.get("hard_errors", []))

        metric_raw = review.get("metric", None)
        metric_val = float(metric_raw) if isinstance(metric_raw, (int, float)) else None
        lower_is_better = bool(review.get("lower_is_better", True))
        maximize = not lower_is_better

        # 只要运行失败、拿不到 metric、或协议不合规，都视作 buggy 节点。
        # 这样的节点仍然保留在树里，但不会进入“优良候选”池，只能走 debug 分支。
        node.is_buggy = (not node.success) or (metric_val is None) or (not node.is_compliant)
        if node.is_buggy:
            node.metric = WorstMetricValue(maximize=maximize)
            node.score = 9999.0 if lower_is_better else -9999.0
        else:
            node.metric = MetricValue(value=metric_val, maximize=maximize)
            node.score = float(metric_val)

        return review

    def step(self, exec_callback: Callable[[str], ExecutionResult]) -> Node:
        """
        执行一轮完整的 AIDE 迭代，对应论文 Algorithm 1 的一次 for-loop。

        流程是：
        1. 用 `search_policy()` 选 action 和 parent；
        2. 调 LLM 生成新代码；
        3. 执行评估；
        4. 写入 Journal；
        5. 打印当前树结构。
        """
        step_index = self.journal.num_nodes + 1
        policy = self.search_policy()
        action: str = policy["action"]
        parent: Optional[Node] = policy["parent"]
        family_hint: Optional[str] = policy.get("family_hint")

        best_before = self.journal.get_best_node()
        best_str = f"{best_before.score:.6f}（{best_before.node_id}）" if best_before else "None"
        parent_str = parent.node_id if parent else "None"
        topk = policy.get("topk", [])
        topk_str = ", ".join([f"{n.node_id}:{n.score:.4f}" for n in topk]) if topk else "-"
        picked_by = policy.get("picked_by", "-")
        family_str = family_hint or "-"

        logger.info(f"\n===== 第 {step_index} 轮 =====")
        logger.info(
            f"策略选择：action={action} | parent={parent_str} | 当前best={best_str} | "
            f"family={family_str} | topk={topk_str} | picked_by={picked_by}"
        )

        logger.info("LLM：开始生成代码...")
        thought, code = self._generate_plan_and_code(action, parent, family_hint=family_hint)
        logger.info(f"LLM：代码生成完成（长度={len(code)}）")

        node_family = family_hint or (parent.family_hint if parent else "")
        node = Node(
            code=code,
            parent=parent,
            stage=action,
            thought=thought,
            family_hint=node_family,
        )
        logger.info(f"节点创建：id={node.node_id} | stage={node.stage}")
        snap_path = self._dump_solution_code(
            step_index=step_index,
            stage_name=node.stage_name,
            node_id=node.node_id,
            code=node.code,
        )
        logger.info(f"代码快照已保存：{snap_path}")

        logger.info("执行：运行 working/solution.py ...")
        result = exec_callback(code)
        logger.info(
            f"执行结果：success={result.success} | 用时={result.execution_time:.2f}s | err={result.error or '-'}"
        )

        if result.output:
            tail = "\n".join(result.output.splitlines()[-30:])
            logger.info("输出末尾（最后 30 行）：\n" + tail)

        logger.info("评审：程序化校验 + 结构化摘要...")
        self.parse_exec_result(node, result)
        self.journal.append(node)

        logger.info(
            f"本轮结果：node={node.node_id} | SCORE={node.score:.6f} | "
            f"buggy={node.is_buggy} | compliant={node.is_compliant}"
        )

        if node.analysis:
            logger.info("评审摘要：" + node.analysis[:600])

        best_now = self.journal.get_best_node()
        best_now_str = f"{best_now.score:.6f}（{best_now.node_id}）" if best_now else "None"
        logger.info(f"当前最优：{best_now_str}")

        self.journal.print_tree()
        self.journal.save(WORKSPACE_DIR)
        return node

    # -------------------- main loop --------------------
    def run(self, max_steps) -> None:
        """
        运行整个搜索过程，并在结束后复跑当前最佳代码。

        论文 Algorithm 1 的最后一步是返回全局最优解；这里除了找到 best node，还会额外：
        - 保存 `best.py`
        - 再跑一次 best code，确保最终 submission 与最优节点一致
        - 打印最终树结构，方便人工复盘
        """
        for _ in range(int(max_steps)):
            self.step(self._execute)

        best = self.journal.get_best_node()
        if best is not None:
            logger.info(f"✅ 最终最优节点：{best.node_id} | SCORE={best.score:.6f}")
            best_path = self._save_best_code(best)
            logger.info(f"最优代码已保存：{best_path}")

            logger.info("复跑最优代码，覆盖最终 submission ...")
            final_result = self._execute(best.code)
            logger.info(
                f"最优代码复跑结果：success={final_result.success} | "
                f"time={final_result.execution_time:.2f}s | err={final_result.error or '-'}"
            )
            if final_result.output:
                tail = "\n".join(final_result.output.splitlines()[-20:])
                logger.info("最优复跑输出末尾（最后 20 行）：\n" + tail)
            final_check = self._programmatic_review(best.code, final_result)
            logger.info(
                "最终 submission 校验："
                f"compliant={final_check['is_compliant']} | "
                f"metric={final_check['metric']} | "
                f"errors={final_check['hard_errors'] or '-'}"
            )

            promoted = self._promote_submission_file()
            if promoted:
                logger.info(f"最终可提交文件：{promoted}")
            else:
                logger.warning("最终未找到 submission.csv，请检查最优代码输出路径。")
        else:
            logger.info("⚠️ 最终没有找到有效（非 buggy）的节点。")

        logger.info("最终树结构：")
        self.journal.print_tree()
