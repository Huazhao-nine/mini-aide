import unittest

from core.metric import MetricValue, WorstMetricValue


class MetricValueTest(unittest.TestCase):
    def test_minimize_metric_compare(self):
        self.assertGreater(MetricValue(0.81, maximize=False), MetricValue(0.89, maximize=False))

    def test_maximize_metric_compare(self):
        self.assertGreater(MetricValue(0.89, maximize=True), MetricValue(0.81, maximize=True))

    def test_worst_metric_is_always_worse(self):
        self.assertGreater(MetricValue(1.0, maximize=False), WorstMetricValue(maximize=False))
        self.assertFalse(WorstMetricValue(maximize=False) > MetricValue(1.0, maximize=False))


if __name__ == "__main__":
    unittest.main()
