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
    analysis: str = ""  # 用于存储 Reviewer 的文字评价
    
    def __post_init__(self):
        if self.parent:
            self.parent.children.append(self)
        self.is_buggy = not self.success

    @property
    def summary(self):
        """
        [只读属性] 用于打印树状图的简报
        """
        # 使用简单的 Emoji 标识状态
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
        self.root: Optional[Node] = None

    def add_node(self, 
                 parent: Optional[Node], 
                 code: str, 
                 thought: str, 
                 stage: str, 
                 exec_result: ExecutionResult) -> Node:
        
        # 1. 尝试正则提取分数 (作为 Reviewer 之前的兜底)
        score = self._extract_score(exec_result.output)
        
        # 2. 创建节点
        node = Node(
            parent=parent,
            stage=stage,
            code=code,
            thought=thought,
            success=exec_result.success,
            output=exec_result.output,
            error=exec_result.error,
            execution_time=exec_result.execution_time,
            score=score if exec_result.success else 0.0
        )
        
        # 3. 注册
        self.nodes.append(node)
        if parent is None:
            self.root = node
            
        # 实时打印简报
        print(f"📝 [Journal] Node Recorded: {node.summary}")
        return node

    def get_best_node(self) -> Optional[Node]:
        successful_nodes = [n for n in self.nodes if n.success]
        if not successful_nodes:
            return None
        return max(successful_nodes, key=lambda n: n.score)

    def get_history_trace(self, node: Node) -> str:
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        
        context_str = "--- Previous History ---\n"
        for i, n in enumerate(path):
            context_str += f"Step {i+1} [{n.stage}]:\n"
            context_str += f"Code Snippet: {n.code[:100]}...\n"
            if not n.success:
                context_str += f"Error: {n.error[:300]}... (truncated)\n"
            else:
                context_str += f"Output Score: {n.score}\n"
                if n.analysis:
                    context_str += f"Review: {n.analysis}\n"
            context_str += "------------------------\n"
        return context_str

    def _extract_score(self, text: str) -> float:
        if not text: return 0.0
        match = re.search(r'(?:score|accuracy|acc|val_acc|rmse).*?(\d+\.\d+)', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return 0.0
    def print_tree(self):
        """
        [优化版] 支持打印多棵树（Forest），显示所有 Draft 分支
        """
        print("\n🌳 === Solution Tree (Forest) ===")
        
        # 1. 找到所有的根节点（即没有父节点的节点）
        roots = [n for n in self.nodes if n.parent is None]
        
        if not roots:
            print("(Empty Tree)")
            return

        # 2. 递归打印函数
        def _print_recursive(node: Node, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + node.summary)
            
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(node.children)
            for i, child in enumerate(node.children):
                _print_recursive(child, new_prefix, i == child_count - 1)

        # 3. 遍历打印所有的根节点
        for i, root in enumerate(roots):
            is_last_root = (i == len(roots) - 1)
            # 这里将多个 Draft 视为同级的树根进行打印
            _print_recursive(root, prefix="", is_last=is_last_root)
            
        print("=======================\n")