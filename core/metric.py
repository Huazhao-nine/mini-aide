"""
统一 metric 的“更优”比较语义。

论文中的 objective function h(s) 可以是 accuracy、AUC、MSE 等任意标量，而且有的指标
越大越好，有的越小越好。为了让 `Journal.get_best_node()` 不关心具体指标方向，这里把
“比较大小”抽象成“谁更优”。
"""

from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Optional


@dataclass
@total_ordering
class MetricValue:
    """
    可比较的指标包装器。

    关键点在于 `>` 的语义被重定义为“更优于”，而不是“数值更大”。
    例如在 MSE 任务里，0.81 > 0.89 会返回 True，因为 0.81 更好。
    """

    value: Optional[float]
    maximize: bool

    def __post_init__(self) -> None:
        if self.value is not None:
            self.value = float(self.value)

    def __gt__(self, other: Any) -> bool:
        # `None` 表示最差（例如 buggy 节点没有有效指标）。
        if not isinstance(other, MetricValue):
            return NotImplemented
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.maximize != other.maximize:
            raise ValueError("Cannot compare metrics with different optimize directions")
        if self.value == other.value:
            return False
        return self.value > other.value if self.maximize else self.value < other.value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return False
        return self.value == other.value and self.maximize == other.maximize

    @property
    def is_worst(self) -> bool:
        return self.value is None

    def __str__(self) -> str:
        direction = "max" if self.maximize else "min"
        val = "None" if self.value is None else f"{self.value:.6f}"
        return f"MetricValue({val}, {direction})"


class WorstMetricValue(MetricValue):
    """始终劣于任何有效指标值，通常用于 buggy / 不合规节点。"""

    def __init__(self, maximize: bool):
        super().__init__(value=None, maximize=maximize)
