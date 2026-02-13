import uuid
import time
import re
from dataclasses import dataclass, field
from typing import List, Optional
from core.interpreter import ExecutionResult

@dataclass
class Node:
    """
    搜索树中的一个节点
    """
    # ---- 核心身份 ----
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stage: str = "draft"
    create_time: float = field(default_factory=time.time)
    
    # ---- 树结构 ----
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)
    
    # ---- 内容 ----
    code: str = ""
    thought: str = ""
    
    # ---- 执行结果 ----
    success: bool = False
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    
    # ---- 评价 ----
    score: float = 0.0
    is_buggy: bool = True
    analysis: str = ""
    
    def __repr__(self):
        return f"Node({self.node_id}, score={self.score})"

    @property
    def summary(self):
        if self.success:
            status = "🟢" 
            metric_str = f"Score: {self.score:.4f}"
        else:
            status = "🔴"
            metric_str = "Failed"
            
        time_str = f"{self.execution_time:.1f}s"
        return f"[{status}] [{self.stage.upper()}] {metric_str} (Time: {time_str}) (ID: {self.node_id})"


class Journal:
    def __init__(self):
        self.nodes: List[Node] = []

    def add_node(self, node: Node):
        """
        [关键修复] 直接接收 Agent 创建好的 Node 对象，确保 ID 一致
        """
        # 1. 注册节点
        self.nodes.append(node)
        
        # 2. 实时打印简报
        print(f"📝 [Journal] Node Recorded: {node.summary}")

    def get_best_node(self) -> Optional[Node]:
        successful_nodes = [n for n in self.nodes if n.success]
        if not successful_nodes:
            return None
        return max(successful_nodes, key=lambda n: n.score)

    def get_history_trace(self, node: Node) -> str:
        # (保持你之前修改好的版本不变)
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        
        context_str = "--- Previous History Trace ---\n"
        full_context_limit = 3 
        
        for i, n in enumerate(path):
            is_recent = (i >= len(path) - full_context_limit)
            context_str += f"\n=== Step {i+1} [{n.stage.upper()}] ===\n"
            
            if n.thought:
                context_str += f"🧠 Plan/Thought:\n{n.thought.strip()}\n"
            else:
                context_str += "🧠 Plan/Thought: (No record)\n"
            
            if is_recent:
                context_str += f"\n💻 Code:\n```python\n{n.code}\n```\n"
                if not n.success:
                    error_snippet = n.error[-1000:] if len(n.error) > 1000 else n.error
                    context_str += f"\n❌ Error Log:\n...{error_snippet}\n"
                else:
                    context_str += f"\n✅ Execution Output:\nScore: {n.score:.4f}\n"
                    if n.analysis:
                        context_str += f"Review Summary: {n.analysis}\n"
            else:
                context_str += "\n💻 Code: [Hidden to save tokens]\n"
                if not n.success:
                    context_str += "❌ Status: FAILED (Execution Error)\n"
                else:
                    context_str += f"✅ Status: SUCCESS | Score: {n.score:.4f}\n"
                    if n.analysis:
                        summary = n.analysis.split('。')[0]
                        context_str += f"Review: {summary}...\n"
            
            context_str += "----------------------------------\n"
            
        return context_str

    def print_tree(self):
        print("\n🌳 === Solution Tree (Forest) ===")
        # 找到所有根节点（parent 为空的节点）
        roots = [n for n in self.nodes if n.parent is None]
        
        if not roots:
            print("(Empty Tree)")
            return

        def _print_recursive(node: Node, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + node.summary)
            
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(node.children)
            for i, child in enumerate(node.children):
                _print_recursive(child, new_prefix, i == child_count - 1)

        for i, root in enumerate(roots):
            is_last_root = (i == len(roots) - 1)
            _print_recursive(root, prefix="", is_last=is_last_root)
            
        print("=======================\n")