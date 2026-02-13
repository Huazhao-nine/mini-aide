from config import WORKSPACE_DIR,num_drafts,debug_prob,max_debug_depth,timeout

class AgentPrompt:
    def __init__(self, task_desc):
        self.task_desc = task_desc
        self.timeout = timeout
        self.workspace_dir = WORKSPACE_DIR

    # ==========================================
    # 📚 1. 通用组件 (Common Components)
    # ==========================================

    @property
    def _prompt_resp_fmt(self):
        return {
            "回复格式 (Response format)": (
                "你的回复应该是用自然语言对你建议的解决方案的简要大纲/草图，"
                "紧接着是一个实现该解决方案并打印评估指标的 "
                "Markdown 代码块（包裹在 ```python ``` 中）。你的回复中不应包含额外的标题或文本。"
                "只需解决方案的简要大纲/草图，后跟一个换行符，然后是 Markdown 代码块。"
            )
        }

    @property
    def _prompt_impl_guideline(self):
        return {
            "实现指南 (Implementation guideline)": [
                "代码应**实现建议的解决方案**并**打印在留出验证集 (hold-out validation set) 上计算的评估指标值**。",
                "代码应该是一个单文件 Python 程序，自包含且可以原样执行。",
                "代码的任何部分都不应被跳过，不要在脚本完成之前终止。",
                f"注意代码的运行时间，它应在{self.timeout}秒内完成。",
                "所有提供的输入数据都存储在 './input' 目录下。",
                "如果你为该任务提供了测试数据，请按照任务描述中的说明，将测试预测结果保存到 `submission.csv` 文件中。"
                "这非常重要，因为该文件用于评分/评估。不要忘记 `submission.csv` 文件！",
                f"你也可以使用 `{self.workspace_dir}` 目录来存储你的代码需要创建的任何临时文件。",
                "评估应基于 K 折交叉验证，但前提是这对当前任务是合适的评估方法，并且 K 值根据具体任务来选择。"
            ]
        }

    @property
    def _prompt_environment(self):
        return {
            "运行环境 (Installed Packages)": (
                "你的解决方案可以使用任何相关的机器学习包，例如：`numpy`, `pandas`, `scikit-learn`, "
                "`statsmodels`, `xgboost`, `lightGBM`, `torch`, `torchvision`, `torch-geometric`, "
                "`bayesian-optimization`, `timm`。请随意使用其他包（所有包都已安装！）。"
                "对于神经网络，我们建议使用 PyTorch 而不是 TensorFlow。"
            )
        }

    # ==========================================
    # 🎯 2. 阶段一：起草 (Draft)
    # ==========================================

    def get_draft_prompt(self, history_trace, data_preview=None):
        prompt = {
            "系统消息 (Introduction)": "你是一位参加竞赛的 Kaggle Grandmaster。为了赢得这次比赛，你需要想出一个出色且有创意的解决方案计划，然后用 Python 实现这个解决方案。我们现在提供任务描述。",
            "任务描述 (Task description)": self.task_desc,
            "记忆 (Memory)": history_trace,
            "指令 (Instructions)": {
                "解决方案草图指南 (Solution sketch guideline)": [
                    "第一个解决方案设计应该相对简单，不包含集成 (ensembling) 或超参数优化。",
                    "在提出设计时要考虑历史节点的做法，不要提出相同的建模解决方案，但保持评估方法一致。",
                    "解决方案草图应为 3-5 句话。",
                    "提出一个对该任务合理的评估指标。",
                    "不要建议做探索性数据分析 (EDA)。",
                    "数据已准备好并在 `./input` 目录下可用。无需解压任何文件。"
                ]
            }
        }
        # 整合通用组件
        prompt["指令 (Instructions)"].update(self._prompt_resp_fmt)
        prompt["指令 (Instructions)"]["实现指南 (Implementation guideline)"] = self._prompt_impl_guideline["实现指南 (Implementation guideline)"]
        prompt["指令 (Instructions)"].update(self._prompt_environment)
        
        return prompt

    # ==========================================
    # 📈 3. 阶段二：改进 (Improve)
    # ==========================================

    def get_improve_prompt(self, journal_summary, parent_node_code):
        prompt = {
            "系统消息 (Introduction)": (
                "你是一位参加竞赛的 Kaggle Grandmaster。下面提供了一个先前开发的解决方案，你应该对其进行改进以进一步提高（测试时）性能。"
                "为此，你应该首先用自然语言概述如何改进解决方案的简要计划，然后基于提供的先前解决方案用 Python 实现此改进。"
            ),
            "任务描述 (Task description)": self.task_desc,
            "记忆 (Memory)": journal_summary,
            "先前解决方案 (Previous solution)": {
                "代码 (Code)": f"```python\n{parent_node_code}\n```"
            },
            "指令 (Instructions)": {
                "解决方案改进草图指南 (Solution improvement sketch guideline)": [
                    "解决方案草图应该是对如何改进先前解决方案的简要自然语言描述。",
                    "你应该非常具体，并且只应提出一个可执行的改进。",
                    "这个改进应该是原子性的 (atomic)，以便我们可以通过实验评估所提出更改的效果。",
                    "在提出改进时要考虑历史节点的做法。",
                    "解决方案草图应为 3-5 句话。",
                    "不要建议做探索性数据分析 (EDA)。"
                ]
            }
        }
        
        # 整合通用组件
        prompt["指令 (Instructions)"].update(self._prompt_resp_fmt)
        prompt["指令 (Instructions)"]["实现指南 (Implementation guideline)"] = self._prompt_impl_guideline["实现指南 (Implementation guideline)"]
        prompt["指令 (Instructions)"].update(self._prompt_environment)
        
        return prompt

    # ==========================================
    # 🛠️ 4. 阶段三：调试 (Debug)
    # ==========================================

    def get_debug_prompt(self, parent_node_code, term_out, data_preview=None):
        prompt = {
            "系统消息 (Introduction)": (
                "你是一位参加竞赛的 Kaggle Grandmaster。你之前的解决方案有一个 Bug，因此基于以下信息，你应该对其进行修改以修复此 Bug。"
                "你的回复应该是自然语言的实现大纲，紧接着是一个实现 Bug 修复/解决方案的 Markdown 代码块。"
            ),
            "任务描述 (Task description)": self.task_desc,
            "先前(有Bug的)实现 (Previous (buggy) implementation)": f"```python\n{parent_node_code}\n```",
            "执行输出 (Execution output)": f"```\n{term_out}\n```",
            "指令 (Instructions)": {
                "Bug修复草图指南 (Bugfix improvement sketch guideline)": [
                    "你应该写一段简短的自然语言描述（3-5 句话），说明如何修复先前实现中的问题。",
                    "不要建议做探索性数据分析 (EDA)。"
                ]
            }
        }
        if data_preview:
            prompt["数据概览 (Data Overview)"] = data_preview
            
        # 整合通用组件
        prompt["指令 (Instructions)"].update(self._prompt_resp_fmt)
        prompt["指令 (Instructions)"]["实现指南 (Implementation guideline)"] = self._prompt_impl_guideline["实现指南 (Implementation guideline)"]
        prompt["指令 (Instructions)"].update(self._prompt_environment)
        
        return prompt

# ==========================================
    # ⚖️ 5. 结果审查 (Result Review)
    # ==========================================

    def get_review_prompt(self, code, term_out):
        return {
            "系统消息 (Introduction)": "你是一位参加竞赛的 Kaggle Grandmaster。你已经编写了代码来解决此任务，现在需要评估代码执行的输出。你应该确定是否存在任何 Bug，并报告实证发现。",
            "任务描述 (Task description)": self.task_desc,
            "实现代码 (Implementation)": f"```python\n{code}\n```",
            "执行输出 (Execution output)": f"```\n{term_out}\n```",
            "裁判任务": [
                "1. 提取分数: 寻找输出中类似 'Score: 0.1234' 或 'MSE: ...' 的行。",
                "2. 判定 Bug: 如果出现 Traceback/Error 或 Score 无效（NaN/0.0/未找到），则 is_bug=True。",
                "3. 总结: 描述实验结果或提出修复建议。"
            ],
            # ✅ 新增：在这里定义输出格式，而不是在 Agent 里拼接
            "输出格式 (Output Format)": (
                "请严格只输出一个 JSON 对象，不要包含 Markdown 格式（如 ```json）。格式如下：\n"
                '{"is_bug": boolean, "score": float, "summary": "string"}'
            )
        }