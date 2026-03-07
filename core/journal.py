"""
维护 solution tree 的数据结构。

论文里所有历史候选解都会被保存到一棵树 T 中，节点是脚本，边表示“从父方案出发做了一次
 draft / debug / improve”。本文件就是这个树结构的本地实现：
- `Node` 表示一个候选脚本及其运行结果；
- `Journal` 负责追加节点、找最优节点、生成历史摘要、保存整棵树。
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.interpreter import ExecutionResult
from core.metric import MetricValue, WorstMetricValue

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class Node:
    # 基本树结构信息：`parent/children` 共同定义了论文中的 solution tree。
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stage: str = "draft"
    step: Optional[int] = None
    create_time: float = field(default_factory=time.time)
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)

    # LLM outputs：既保存生成的脚本，也保存“这次想做什么”的高层说明。
    code: str = ""
    thought: str = ""
    family_hint: str = ""

    # Execution result (structured)：执行器返回的原始结构化结果。
    term_out: List[str] = field(default_factory=list)
    exec_time: float = 0.0
    exc_type: Optional[str] = None
    exc_info: Optional[Dict[str, Any]] = None
    exc_stack: Optional[List[Any]] = None

    # Compatibility execution fields：给当前 mini-aide 其余模块直接消费的人类友好字段。
    success: bool = False
    output: str = ""
    error: str = ""
    execution_time: float = 0.0

    # Evaluation：评审后得到的“是否有效 / 指标多少 / 下一步建议”。
    score: float = 9999.0
    metric: MetricValue = field(default_factory=lambda: WorstMetricValue(maximize=False))
    is_buggy: bool = True
    is_compliant: bool = False
    compliance_errors: List[str] = field(default_factory=list)
    analysis: str = ""

    def __post_init__(self) -> None:
        # 创建节点时自动把它挂到父节点下面，保持树结构一致。
        if self.parent is not None and self not in self.parent.children:
            self.parent.children.append(self)

    def __repr__(self) -> str:
        return f"Node({self.node_id}, score={self.score})"

    @property
    def plan(self) -> str:
        return self.thought

    @plan.setter
    def plan(self, value: str) -> None:
        self.thought = value or ""

    @property
    def stage_name(self) -> str:
        # 对外展示时，节点语义由父节点状态决定：
        # 根节点是 draft；从 buggy 节点延伸出来的是 debug；否则是 improve。
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def debug_depth(self) -> int:
        # 论文中 debug 分支通常会限制深度，避免在坏分支上无限修补。
        if self.stage_name != "debug" or self.parent is None:
            return 0
        return self.parent.debug_depth + 1

    def absorb_exec_result(self, exec_result: ExecutionResult) -> None:
        # 把执行器结果灌进节点，后续 review / 排序 / 历史摘要都依赖这些字段。
        self.term_out = list(exec_result.term_out or [])
        self.exec_time = float(exec_result.exec_time or 0.0)
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

        self.success = exec_result.success
        self.output = exec_result.output
        self.error = exec_result.error
        self.execution_time = exec_result.execution_time

    @property
    def summary(self) -> str:
        # 单行摘要主要服务于日志和树打印，便于观察每个节点的状态变化。
        if self.success and not self.is_buggy:
            status = "OK"
            metric_str = f"SCORE: {self.score:.6f}"
        elif self.success:
            status = "FLAG"
            metric_str = f"Flagged score: {self.score:.6f}"
            if self.compliance_errors:
                metric_str += " (non-compliant)"
        else:
            status = "FAIL"
            metric_str = "Failed"
        time_str = f"{self.execution_time:.1f}s"
        return f"[{status}] [{self.stage_name.upper()}] {metric_str} (Time: {time_str}) (ID: {self.node_id})"


class Journal:
    """
    solution tree 的容器。

    它除了保存节点，还承担论文中 summarization operator 的一部分职责：把历史结果压缩成
    可放进 prompt 的文本记忆，避免把所有旧日志直接塞进上下文。
    """
    def __init__(self):
        self.nodes: List[Node] = []

    def append(self, node: Node) -> Node:
        # 按追加顺序为节点分配 step，便于还原搜索轨迹。
        if node.step is None:
            node.step = len(self.nodes)
        self.nodes.append(node)
        print(f"[Journal] Node Recorded: {node.summary}")
        return node

    def add_node(self, node: Optional[Node] = None, **kwargs) -> Node:
        if node is not None:
            return self.append(node)

        code: str = kwargs.get("code", "")
        parent: Optional[Node] = kwargs.get("parent", None)
        stage: str = kwargs.get("stage", kwargs.get("action", "draft"))
        thought: str = kwargs.get("thought", "")
        family_hint: str = kwargs.get("family_hint", "")

        n = Node(stage=stage, code=code, thought=thought, family_hint=family_hint, parent=parent)
        return self.append(n)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def draft_nodes(self) -> List[Node]:
        # 根节点集合，对应 search policy 里“是否已经有足够初始解”的判断。
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        # 只有非 buggy 节点才有资格进入 improve 候选池。
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> List[MetricValue]:
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        # 这里依赖 `MetricValue` 的“更优”比较语义，而不是简单数值大小。
        pool = self.good_nodes if only_good else self.nodes
        if not pool:
            return None
        return max(pool, key=lambda n: n.metric)

    def best_node(self) -> Optional[Node]:
        return self.get_best_node()

    def sample_buggy_leaf(self) -> Optional[Node]:
        # debug 只从 buggy leaf 中抽样，避免对已有后继的错误中间节点重复修补。
        candidates = [n for n in self.nodes if n.is_buggy and n.is_leaf]
        if not candidates:
            return None
        import random

        return random.choice(candidates)

    def generate_summary(self, include_code: bool = False, max_items: int = 8) -> str:
        """
        生成全局历史摘要。

        这就是论文中的 Σ(T) 的一个具体实现：不是把所有节点全量拼接，而是只保留若干表现最好的
        非 buggy 节点，浓缩它们的设计意图、结果和 metric。
        """
        goods = [n for n in self.good_nodes if n.success]
        goods = sorted(goods, key=lambda n: n.metric, reverse=True)[:max_items]
        if not goods:
            return "(no successful non-buggy runs yet)"

        parts: List[str] = []
        for n in goods:
            block: List[str] = []
            block.append(f"Design: {n.plan.strip() or '(no plan)'}")
            if include_code:
                block.append(f"Code:\n{n.code}")
            block.append(f"Results: {n.analysis.strip() or '(no analysis)'}")
            metric_val = "None" if n.metric.value is None else f"{n.metric.value:.6f}"
            block.append(f"Validation Metric: {metric_val}")
            parts.append("\n".join(block))
        return "\n-------------------------------\n".join(parts)

    def build_memory(self, node: Optional[Node] = None) -> str:
        """
        生成给 LLM 使用的“记忆”文本。

        由两部分组成：
        1. best-so-far：全局最优经验，帮助模型避免重复尝试已证伪方向；
        2. local trace：当前节点往上的局部祖先链，帮助模型理解这一分支最近做过什么。
        """
        global_sum = self.generate_summary(include_code=False)
        if node is None:
            return global_sum

        trace: List[str] = []
        cur = node
        hop = 0
        while cur is not None and hop < 5:
            trace.append(f"- {cur.summary}\n  findings: {cur.analysis.strip() or '(none)'}")
            cur = cur.parent
            hop += 1
        trace_str = "\n".join(trace)
        return (
            "## Best-so-far notes\n"
            f"{global_sum}\n\n"
            "## Local trace (most recent first)\n"
            f"{trace_str}"
        )

    def get_history_trace(self, node: Optional[Node] = None) -> str:
        return self.build_memory(node)

    def save(self, workspace_dir: str, filename: str = "journal.jsonl") -> None:
        # 持久化整个树，答辩或复盘时可以直接查看每轮节点的代码、错误和评分。
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            path = os.path.join(workspace_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                for n in self.nodes:
                    d = {
                        "node_id": n.node_id,
                        "stage": n.stage,
                        "stage_name": n.stage_name,
                        "step": n.step,
                        "create_time": n.create_time,
                        "parent_id": n.parent.node_id if n.parent else None,
                        "children_ids": [c.node_id for c in n.children],
                        "code": n.code,
                        "thought": n.thought,
                        "family_hint": n.family_hint,
                        "term_out": n.term_out,
                        "exec_time": n.exec_time,
                        "exc_type": n.exc_type,
                        "exc_info": n.exc_info,
                        "exc_stack": n.exc_stack,
                        "success": n.success,
                        "output": n.output,
                        "error": n.error,
                        "execution_time": n.execution_time,
                        "score": n.score,
                        "metric": {
                            "value": n.metric.value,
                            "maximize": n.metric.maximize,
                        },
                        "is_buggy": n.is_buggy,
                        "is_compliant": n.is_compliant,
                        "compliance_errors": n.compliance_errors,
                        "analysis": n.analysis,
                    }
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception:
            return

    def print_tree(self) -> None:
        # 以 ASCII 树打印当前 forest，直观看每个分支是 draft / debug / improve 的哪条路径。
        logger.info("\n=== Solution Tree (Forest) ===")
        roots = [n for n in self.nodes if n.parent is None]
        if not roots:
            logger.info("(Empty Tree)")
            return

        def _print_recursive(node: Node, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            logger.info(prefix + connector + node.summary)
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(node.children)
            for i, child in enumerate(node.children):
                _print_recursive(child, new_prefix, i == child_count - 1)

        for i, root in enumerate(roots):
            _print_recursive(root, prefix="", is_last=(i == len(roots) - 1))
        logger.info("=======================\n")
