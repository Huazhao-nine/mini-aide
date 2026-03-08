"""
`core.metric` 的语义回归测试。

这些测试不是论文的一部分，但它们在工程上保护了论文中的 objective function 比较语义：
- 不同任务可以是 maximize 或 minimize；
- buggy 节点应该始终比有效节点差；
- `Journal.get_best_node()` 依赖这里的排序规则。

复试时可以说：这是为了确保“树上谁是当前最优节点”不会因为工程修改而被悄悄破坏。
"""

import unittest

from core.metric import MetricValue, WorstMetricValue


class MetricValueTest(unittest.TestCase):
    """验证“更优”而不是“数值更大”的比较语义。"""

    def test_minimize_metric_compare(self):
        self.assertGreater(MetricValue(0.81, maximize=False), MetricValue(0.89, maximize=False))

    def test_maximize_metric_compare(self):
        self.assertGreater(MetricValue(0.89, maximize=True), MetricValue(0.81, maximize=True))

    def test_worst_metric_is_always_worse(self):
        self.assertGreater(MetricValue(1.0, maximize=False), WorstMetricValue(maximize=False))
        self.assertFalse(WorstMetricValue(maximize=False) > MetricValue(1.0, maximize=False))


if __name__ == "__main__":
    unittest.main()
