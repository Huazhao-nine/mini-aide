import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径 
MODEL_PATH = "/home/huazhao/DL-HW/ML-2025/Model/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# 代码执行沙盒
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

num_drafts = 3
debug_prob = 0.5
max_debug_depth = 3
max_steps=20
timeout=3600