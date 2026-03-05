from config import WORKSPACE_DIR, timeout


class AgentPrompt:
    def __init__(self, task_desc):
        self.task_desc = task_desc
        self.timeout = timeout
        self.workspace_dir = WORKSPACE_DIR

    @property
    def _resp_fmt(self):
        return {
            "回复格式": [
                "先写 3-5 句简短方案说明。",
                "然后输出且只输出一个 ```python``` 代码块。",
                "不要输出第二个代码块，不要省略代码。",
            ]
        }

    @property
    def _protocol(self):
        return {
            "输出协议": [
                "脚本最后必须打印：FINAL_MSE=<数值>。",
                "可选打印：FINAL_INFO=<json字符串>，用于记录模型家族、CV、seed、feature_count 等。",
                "如果原方案已生成 submission.csv，除非本轮明确针对提交逻辑，否则不要破坏它。",
            ]
        }

    @property
    def _env(self):
        return {
            "运行环境": [
                "你输出的是单文件 Python 脚本，本地直接执行。",
                f"工作目录：{self.workspace_dir}",
                f"时间限制：约 {self.timeout} 秒。",
                "可使用 numpy/pandas/scikit-learn/lightgbm/xgboost/torch。若任务要求 DNN，优先 PyTorch。",
            ]
        }

    @property
    def _general_rules(self):
        return {
            "通用约束": [
                "优先保留父方案已验证有效的部分。",
                "不要做 EDA、交互式可视化或长篇分析。",
                "避免数据泄漏，注意 split/CV/scaler 的 fit-transform 边界。",
                "一次只改一个主要因素，保证实验可比较。",
                "尽量固定 random/numpy/torch 的随机种子。",
            ]
        }

    def _base_prompt(self):
        base = {}
        base.update(self._resp_fmt)
        base.update(self._protocol)
        base.update(self._env)
        base.update(self._general_rules)
        return base

    def get_draft_prompt(self, history, data_preview=None, force_new_family: bool = False):
        draft_rules = [
            "先写出稳定、可运行、评估逻辑正确的 baseline。",
            "若任务描述明确要求 DNN/MLP 主线，则优先使用 PyTorch MLP，而不是树模型。",
            "Draft 阶段先建立一条强而稳的单模型主线，不要一开始做 stacking 或大规模超参搜索。",
            "必须保证 metric、CV、submission、特征边界都正确。",
        ]
        if force_new_family:
            draft_rules.append("当前搜索已停滞：请换一个与最近最佳方案明显不同的 DNN 变体，例如更稳的 MLP 结构、不同归一化/激活/正则主线，但仍保持简单。")

        prompt = {
            "系统消息": "你是一个通用机器学习/深度学习竞赛代理。当前任务优先构造深度学习可运行强基线。",
            "任务描述": self.task_desc,
            "精简历史摘要": history,
            "Draft要求": draft_rules,
        }
        # if data_preview:
            # prompt["数据概览"] = data_preview
        prompt.update(self._base_prompt())
        return prompt

    def get_improve_prompt(self, journal_summary, parent_node_code, change_type="feature"):
        hard_templates = {
            "feature": [
                "只允许改特征处理或输入表示；不要改模型家族、CV 主干、训练主干。",
                "如果任务中禁止对 tested_positive_* 派生，则必须遵守。",
                "优先尝试安全模板：非 tested_positive_* 的跨天差分、比例、mean/std/min/max，其它部分保持不变。",
            ],
            "model": [
                "只允许改一个主要模型因素；不要同时改特征、CV、训练流程。",
                "在 DNN 主线内做小改：层宽、层数、激活、BN/LayerNorm、dropout、残差（若实现很简单）。",
                "不要突然切到树模型，除非任务描述没有 DNN 偏好且历史已明确 DNN 全线失败。",
            ],
            "training": [
                "只允许改优化器、学习率、batch size、epoch、scheduler、patience、gradient clipping、seed 等训练细节。",
                "不要改特征工程、CV 主干、模型结构。",
            ],
            "regularization": [
                "只允许改一个主要正则化因素，如 dropout、weight decay、早停、label clipping、目标缩放处理等。",
                "不要顺带重构模型或验证方案。",
            ],
            "validation": [
                "只允许改 split/CV/OOF 聚合/metric 监控方式；不要改模型主干和特征工程。",
                "只有在当前验证不稳或疑似泄漏时才值得动这类改动。",
            ],
            "submission": [
                "只允许改测试集推理、fold 预测聚合、列对齐、后处理、导出格式等提交相关逻辑。",
                "不要改 local validation 主干，不要改模型或主要特征。",
            ],
        }

        improve_rules = [
            "只提出并实现一个 actionable improvement。",
            "必须保留 FINAL_MSE 协议。",
            "如果父方案已有正确评分/提交主干，优先保留。",
            "不要做与本轮改动无关的大改写。",
            "不要复述历史中已经明确失败的方向。",
        ]

        prompt = {
            "系统消息": "你要基于一个已可运行父方案做一次原子化改进。只做一个明确、可归因、可比较的改动。",
            "任务描述": self.task_desc,
            "精简历史经验": journal_summary,
            "父方案代码": f"```python\n{parent_node_code}\n```",
            "本轮改动类型": change_type,
            "Improve要求": improve_rules,
            "本轮硬约束": hard_templates.get(change_type, hard_templates["feature"]),
        }
        prompt.update(self._base_prompt())
        return prompt

    def get_debug_prompt(self, parent_node_code, term_out, data_preview=None):
        prompt = {
            "系统消息": "你之前的代码运行失败了。请做最小限度修复，优先恢复可运行性、评分协议与已有有效逻辑。",
            "任务描述": self.task_desc,
            "先前实现": f"```python\n{parent_node_code}\n```",
            "执行输出": f"```\n{term_out}\n```",
            "Debug要求": [
                "只修当前 bug，不要顺手重构整套方案。",
                "若是列名/shape/类型/协议错误，直接修相应位置。",
                "若父方案已有正确评估/提交主干，调试时必须保留。",
            ],
        }
        if data_preview:
            prompt["数据概览"] = data_preview
        prompt.update(self._base_prompt())
        return prompt

    def get_review_prompt(self, code, term_out):
        return {
            "系统消息": "请基于执行输出做极简评审。系统会程序解析 FINAL_MSE，你只需要给一句到四句的有效总结。",
            "任务描述": self.task_desc,
            "实现代码": f"```python\n{code}\n```",
            "执行输出": f"```\n{term_out}\n```",
            "评审任务": [
                "判断这次运行是否有效。",
                "如果有效，说明主要瓶颈。",
                "给出下一步最值得尝试的一类原子化改动。",
            ],
        }
