import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict

# --- Console logging (minimal, no extra deps) ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False
@dataclass
class Node:
    """AIDE-style solution tree node (simplified).

    - stage: draft / improve / debug
    - parent/children: form a forest (solution tree)
    - analysis: high-signal feedback used by memory (crucial for AIDE improve/debug)
    """

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stage: str = "draft"
    create_time: float = field(default_factory=time.time)
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)

    # LLM outputs
    code: str = ""
    thought: str = ""  # optional short plan / intent

    # Execution results
    success: bool = False
    output: str = ""
    error: str = ""
    execution_time: float = 0.0

    # Evaluation
    score: float = 0.0  # MSE
    is_buggy: bool = True
    analysis: str = ""  # 2-3 sentence findings OR bug-fix suggestion

    def __repr__(self) -> str:
        return f"Node({self.node_id}, score={self.score})"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def debug_depth(self) -> int:
        if self.stage != "debug" or self.parent is None:
            return 0
        return self.parent.debug_depth + 1

    @property
    def summary(self) -> str:
        if self.success and not self.is_buggy:
            status = "🟢"
            metric_str = f"MSE: {self.score:.6f}"
        elif self.success:
            status = "🟡"
            metric_str = f"Flagged: {self.score:.6f}"
        else:
            status = "🔴"
            metric_str = "Failed"
        time_str = f"{self.execution_time:.1f}s"
        return f"[{status}] [{self.stage.upper()}] {metric_str} (Time: {time_str}) (ID: {self.node_id})"


class Journal:
    """AIDE-style journal (simplified but interface-aligned).

    핵심:
    - append-only nodes + tree pointers (parent/children)
    - selection helpers: draft_nodes / buggy_nodes / good_nodes / get_best_node
    - memory helpers: generate_summary / build_memory  (对齐原版的“高信号反馈闭环”)
    """

    def __init__(self):
        self.nodes: List[Node] = []

    # -------------------- append --------------------
    def append(self, node: Node) -> Node:
        self.nodes.append(node)
        print(f"📝 [Journal] Node Recorded: {node.summary}")
        return node

    # Backward-compatible:
    # - add_node(node)
    # - add_node(code=..., parent=..., stage=..., thought=...)
    def add_node(self, node: Optional[Node] = None, **kwargs) -> Node:
        if node is not None:
            return self.append(node)

        code: str = kwargs.get("code", "")
        parent: Optional[Node] = kwargs.get("parent", None)
        stage: str = kwargs.get("stage", kwargs.get("action", "draft"))
        thought: str = kwargs.get("thought", "")

        n = Node(stage=stage, code=code, thought=thought, parent=parent)
        if parent is not None:
            parent.children.append(n)

        return self.append(n)

    # -------------------- properties --------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def draft_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        return [n for n in self.nodes if (not n.is_buggy)]

    # -------------------- selection --------------------
    def get_best_node(self) -> Optional[Node]:
        successful_nodes = [n for n in self.nodes if n.success and not n.is_buggy]
        if not successful_nodes:
            return None
        return min(successful_nodes, key=lambda n: n.score)

    # compatibility alias
    def best_node(self) -> Optional[Node]:
        return self.get_best_node()

    def sample_buggy_leaf(self) -> Optional[Node]:
        candidates = [n for n in self.nodes if n.is_buggy and n.is_leaf]
        if not candidates:
            return None
        import random
        return random.choice(candidates)

    # -------------------- memory / summary --------------------
    def generate_summary(self, include_code: bool = False, max_items: int = 8) -> str:
        """AIDE 风格：汇总当前最好的一些非 buggy 节点（给 improve/debug 当记忆）。"""
        goods = [n for n in self.good_nodes if n.success]
        goods = sorted(goods, key=lambda n: n.score)[:max_items]
        if not goods:
            return "（当前还没有成功且非 buggy 的运行）"

        parts: List[str] = []
        for n in goods:
            chunk = []
            if n.thought.strip():
                chunk.append(f"方案要点：{n.thought.strip()}")
            if include_code:
                chunk.append(f"代码：\n{n.code}")
            if n.analysis.strip():
                chunk.append(f"结果/结论：{n.analysis.strip()}")
            chunk.append(f"验证指标（MSE）：{n.score}")
            parts.append("\n".join(chunk))

        return "\n-------------------------------\n".join(parts)

    def build_memory(self, node: Optional[Node] = None) -> str:
        """Memory used by improve/debug prompts: global best + local trace."""
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

    # compatibility
    def get_history_trace(self, node: Optional[Node] = None) -> str:
        return self.build_memory(node)

    # -------------------- persistence (optional) --------------------
    def save(self, workspace_dir: str, filename: str = "journal.jsonl") -> None:
        """Persist a lightweight snapshot for debugging."""
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            path = os.path.join(workspace_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                for n in self.nodes:
                    d: Dict[str, Any] = asdict(n)
                    d["parent_id"] = n.parent.node_id if n.parent else None
                    d["children_ids"] = [c.node_id for c in n.children]
                    d.pop("parent", None)
                    d.pop("children", None)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception:
            return

    def print_tree(self) -> None:
        logger.info("\n🌳 === Solution Tree (Forest) ===")
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