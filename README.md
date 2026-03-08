# mini-aide

A lightweight reproduction of the AIDE paper for automated machine learning code search.

本项目参考论文《AIDE: AI-Driven Exploration in the Space of Code》，目标不是复现某一个固定模型，而是复现论文里把机器学习工程流程转写为“代码空间自动搜索问题”的核心思想。系统会围绕一个任务定义，在代码空间中持续生成、执行、评估、总结并扩展候选解，最终输出可提交的 Kaggle 风格脚本与 submission 文件。

## Project Goal

传统机器学习工程往往依赖人工试错：改特征、换模型、修 bug、看分数、再迭代。本项目把这件事抽象成 AIDE 风格的自动搜索闭环：

- `solution tree`：把每一轮尝试过的代码、分数、错误和经验组织成树结构
- `search policy`：根据当前树状态决定下一步是 `draft`、`debug` 还是 `improve`
- `coding operator`：调用大模型生成新的候选脚本
- `summarization operator`：把历史有效经验压缩成后续轮次可复用的上下文
- `evaluator`：在受控环境中执行代码，提取 `FINAL_SCORE`、校验 submission 契约、生成结构化执行结果

这也是本仓库的重点：复现 AIDE 的方法论，而不是仅仅堆一个 AutoML baseline。

## Paper Mapping

仓库中的核心模块和论文概念对应关系如下：

- `core/journal.py`
  - 对应论文中的 `solution tree`
  - 保存每个节点的代码、父子关系、分数、错误、摘要和元数据
- `core/agent.py`
  - 对应论文中的主循环与 `search policy`
  - 决定 `draft / debug / improve`，并把任务上下文与历史总结拼成 prompt
- `core/interpreter.py`
  - 对应 evaluator `h(s)` 的执行壳
  - 独立子进程执行候选脚本，收集 stdout / stderr / traceback / timeout
- `core/journal.py` 中的 summary 构造
  - 对应论文中的 `Σ(T)`，即对当前搜索树的压缩记忆
- `backend/llm.py`
  - 不属于论文新贡献
  - 只是把上层的 coding / review operator 接到真实大模型 API 上

从复试讲解角度，你可以把项目概括为：

> 我不是在复现一个具体模型，而是在复现 AIDE 论文提出的“围绕 solution tree 做自动代码搜索”的框架，并把它落到了 Kaggle 风格机器学习任务上。

## Repository Structure

```text
mini-aide/
├── backend/                # LLM 后端适配层
├── core/                   # Agent / Journal / Interpreter / Metric 等核心实现
├── run/                    # 各个 benchmark 任务入口
├── input/                  # Kaggle 数据目录（不建议完整提交到 GitHub）
├── paper/                  # 论文 PDF
├── tests/                  # 核心组件测试
├── utils/                  # 配置、工具函数、提示词模板
├── workspace/              # Journal 等运行时状态
├── working_* /             # 各任务独立工作目录
└── README.md
```

主要源码入口：

- `core/agent.py`：AIDE 主循环
- `core/journal.py`：树结构与历史总结
- `core/interpreter.py`：受控执行器
- `utils/config.py`：搜索预算和全局实验配置
- `run/*.py`：不同 Kaggle benchmark 的任务定义入口

## Supported Benchmarks

当前仓库已经配置了 7 个 Kaggle 风格任务入口：

- `run/Titanic_run.py`
- `run/DigitRecognizer_run.py`
- `run/HousePrices_run.py`
- `run/SpaceshipTitanic_run.py`
- `run/OttoGroup_run.py`
- `run/PortoSeguro_run.py`
- `run/ForestCover_run.py`

这些任务覆盖了多种典型场景：

- 表格二分类：`Titanic`, `Spaceship Titanic`
- 表格回归：`House Prices`
- 多分类表格：`Otto Group`, `Forest Cover Type`
- 排序型二分类：`Porto Seguro`
- 轻量视觉分类：`Digit Recognizer`

因此它们比较适合用于论文复现实验中的 benchmark 统计。

## Task Protocol

为了让不同任务共享同一套 evaluator 逻辑，本项目统一使用：

```python
FINAL_SCORE=<number>
```

约束如下：

- 最后一条非空输出必须是 `FINAL_SCORE=<number>`
- `FINAL_SCORE` 统一按“越小越好”比较
- 各任务内部可以自行定义这个数的语义

例如：

- 回归任务：`FINAL_SCORE = 5-fold OOF RMSE / RMSLE / MSE`
- 分类任务：`FINAL_SCORE = 1 - 5-fold OOF accuracy`
- 排序任务：`FINAL_SCORE = 1 - normalized_gini`
- 多分类概率任务：`FINAL_SCORE = 5-fold OOF log loss`

这样做的好处是：

- 不需要改动整套树搜索和 metric 排序逻辑
- 任务切换成本低
- 更适合批量 benchmark

## How It Works

每个任务入口都会先构造一份静态任务描述，然后把搜索交给 `Agent`：

1. 读取数据并做 `sanity_check`
2. 生成数据概览、评估协议、防泄漏规则和 submission 契约
3. `Agent` 基于当前解树决定本轮是 `draft`、`debug` 还是 `improve`
4. LLM 生成新的完整 Python 脚本
5. `Interpreter` 在受控环境中执行脚本
6. 程序化提取 `FINAL_SCORE`，并检查 submission 文件是否符合任务要求
7. `Journal` 记录该轮代码、分数、错误、摘要与经验
8. 进入下一轮搜索，直到预算耗尽

从实现上看，这是一套“自动化 trial-and-error engine”。

## Environment

建议环境：

- Python `3.10+`
- Linux / WSL / macOS
- 已安装 Kaggle 常见机器学习依赖

常用依赖至少包括：

```bash
pip install pandas numpy scikit-learn openai
```

对于不同任务，建议额外准备：

```bash
pip install torch torchvision
pip install xgboost lightgbm catboost
pip install llama-cpp-python
```

说明：

- `openai` 用于当前默认在线 LLM 后端
- `torch` 主要用于 `Digit Recognizer` 等深度学习任务
- `xgboost / lightgbm / catboost` 不是所有任务都必须，但会显著提升 leaderboard 竞争力
- `llama-cpp-python` 仅在你启用本地模型后端时需要

## LLM Backend Configuration

当前默认在线后端在 `backend/llm.py`，通过环境变量配置：

```bash
export DEEPSEEK_API_KEY=your_api_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-reasoner
```

如果你要改为其他模型供应商，可以保留 `core/` 不动，只替换 `backend/llm.py` 的实现。

## Search Configuration

默认搜索预算在 `utils/config.py` 中配置，当前主要参数包括：

- `num_drafts = 4`
- `max_steps = 24`，随后再加上 `num_drafts`
- `top_k_candidates = 6`
- `debug_prob = 0.35`
- `max_debug_depth = 2`
- `draft_families = "tree,linear,nn"`

这些参数对应的是：

- 初始根节点数量
- 总改进轮数
- improve 阶段的候选选择范围
- 遇到 buggy 节点时切换 debug 的概率
- 调试分支的最大深度

## Data Layout

推荐的数据目录形式如下：

```text
input/
├── titanic/
├── digit_recognizer/
├── house_prices/
├── spaceship_titanic/
├── otto_group/
├── porto_seguro/
└── forest_cover_type/
```

每个任务的列名、submission 格式、CV 协议和防泄漏规则都写在对应的 `run/*.py` 里。

注意：

- GitHub 仓库通常不建议提交完整数据集
- 如果你在本地调整了数据目录，请同步修改对应 `run/*.py` 中的路径配置
- 以脚本内的 `train_path / test_path / submission_path` 为最终准则

## Usage

运行某个 benchmark 的方式如下：

```bash
python run/Titanic_run.py
python run/DigitRecognizer_run.py
python run/HousePrices_run.py
python run/SpaceshipTitanic_run.py
python run/OttoGroup_run.py
python run/PortoSeguro_run.py
python run/ForestCover_run.py
```

典型流程：

1. 选择一个任务入口
2. 准备对应 Kaggle 数据
3. 配置 LLM API
4. 启动搜索
5. 查看 `working_<task>/` 下生成的代码、best 脚本和 submission
6. 把 submission 上传到 Kaggle，记录 leaderboard 结果

## Outputs

每个任务都会有独立工作目录，例如：

```text
working_titanic/
├── best.py
├── solution/
│   ├── step1-draft-xxxx.py
│   ├── step2-debug-xxxx.py
│   └── stepN-improve-xxxx.py
└── working/
    └── submission-titanic.csv
```

常见运行产物：

- `solution/`：每一步生成的候选代码快照
- `best.py`：当前任务的最优脚本
- `working/submission-*.csv`：最终 Kaggle 提交文件
- `workspace/journal.jsonl`：搜索树节点日志
- `logs/*.log`：运行日志

这套目录设计的目的，是让不同任务互不干扰，方便后期做 benchmark 统计与复盘。

## Engineering Constraints

为了让生成代码更稳定，本项目在 prompt 和执行协议里做了几条重要约束：

- 所有任务必须输出 `FINAL_SCORE`
- 最后一条非空输出必须是 `FINAL_SCORE=<number>`
- submission 必须写到任务指定路径，不能随意写到共享目录
- 生成代码默认禁止 `joblib/loky` 多进程并行
- sklearn 相关 `n_jobs` 默认要求为 `1`
- 必须遵守 fold-safe preprocessing，避免 CV 泄漏
- 若当前方案已有效，优先做原子改动，而不是重写整套 pipeline

这些约束不是论文的原文内容，而是本项目为了可复现 benchmark 做的工程化增强。

## Testing

核心模块测试位于 `tests/`，可以直接运行：

```bash
PYTHONPATH=. pytest tests/test_metric.py tests/test_journal.py tests/test_interpreter.py
```

如果只做基本语法检查：

```bash
python -m py_compile core/*.py backend/*.py utils/*.py run/*.py
```

## Reproducibility and Benchmarking

如果你想把这个项目用于论文复试、项目展示或 GitHub 展示，建议按下面的口径组织结果：

- 本地 OOF `FINAL_SCORE`
- Kaggle public leaderboard score
- 运行轮数和搜索预算
- 最佳方案的模型族与关键改进点
- 不同任务间的平均收益与稳定性

你可以把结果汇总成一张表，例如：

| Task | Local Metric | Kaggle Score | Search Steps | Best Strategy |
| --- | --- | --- | --- | --- |
| Titanic | 1 - OOF Accuracy | TBD | TBD | Tree / Linear + feature engineering |
| Digit Recognizer | 1 - OOF Accuracy | TBD | TBD | Small CNN + BN + dropout |
| House Prices | OOF RMSLE | TBD | TBD | Strong tabular preprocessing + boosting |

## Limitations

这个仓库是对论文思想的工程化复现，不是论文作者的官方实现，因此有一些边界：

- 搜索策略目前是硬编码的启发式策略，不是完全学习式 policy
- 任务上下文主要依赖手工整理的 task prompt
- 真正的 leaderboard 表现仍依赖底层模型能力和 API 稳定性
- 不同任务的最佳模型族并不相同，无法靠单一 prompt 一次性最优
- 大规模图像任务和复杂集成任务还可以继续扩展

## Why This Project Matters

这个项目的价值不在于“又做了一个 Titanic 分类器”，而在于：

- 它把论文中的抽象搜索框架具体落地到了可运行代码系统
- 它能跨任务复用同一套自动试错主循环
- 它天然适合做 benchmark、ablation 和错误分析
- 它展示了“LLM 作为机器学习工程搜索器”的工程可行性

如果你把它用于研究生复试，一个很自然的表述是：

> 我的工作重点不是复现某个单点模型，而是把 AIDE 论文中 solution tree、search policy、summarization operator 和 coding operator 这四个核心机制，在 Kaggle 风格任务上做成了一个可跑 benchmark 的系统。

## Paper

- 论文 PDF：`paper/2502.13138v1.pdf`

## License

如果你准备公开发布，建议补充一个明确的开源许可证，例如 `MIT` 或 `Apache-2.0`。

