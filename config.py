import os

# 项目根目录。后续所有默认路径都相对于它构造。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 本地模型路径。当前主流程更多使用在线 API，但这里保留了本地模型配置入口。
MODEL_PATH = "/home/huazhao/DL-HW/ML-2025/Model/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# 代码执行沙盒目录。Agent 生成的脚本、journal、submission 等文件都会落在相关工作目录里。
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

# AIDE 搜索超参数：
# - num_drafts: 初始根节点数量，对应论文 search policy 中的 drafting 阶段
# - debug_prob: 有 buggy leaf 时，切到 debug 的概率
# - max_debug_depth: 单条错误分支允许连续修复的最大深度
# - max_steps: improve/debug 的总轮数（下方会再加上 num_drafts，得到完整循环次数）
# - timeout: 单个候选脚本允许的最长执行时间
num_drafts = 3
debug_prob = 0.5
max_debug_depth = 3
max_steps=20
timeout=3600

# 搜索策略增强：
# - top_k_candidates: improve 时只在前 k 个较优节点中选父节点
# - explore_epsilon: 以一定概率从 top-k 中探索“子节点较少”的分支，避免过早贪心
# - draft_families: 初始草稿阶段希望覆盖的模型家族
top_k_candidates = 4          # improve 时只在 top-k 节点中选 parent
explore_epsilon = 0.35        # 以该概率随机探索 top-k，否则选当前最优
draft_families = "linear,tree,nn"  # draft 阶段家族覆盖顺序（逗号分隔）

# 兼容性开关：为以后扩展不同解释器/指标后端预留接口。
INTERPRETER_MODE = os.getenv("INTERPRETER_MODE", "process").strip().lower()
METRIC_MODE = os.getenv("METRIC_MODE", "generic").strip().lower()

# 总搜索轮数 = 若干 root drafts + 之后的 improve/debug 轮次。
max_steps = max_steps + num_drafts
