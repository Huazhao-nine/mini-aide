import unittest

from core.journal import Journal, Node
from core.metric import MetricValue, WorstMetricValue


class JournalTest(unittest.TestCase):
    def test_get_best_node_uses_metric(self):
        journal = Journal()

        n1 = Node(code="a")
        n1.success = True
        n1.is_buggy = False
        n1.metric = MetricValue(0.89, maximize=False)
        n1.score = 0.89
        journal.append(n1)

        n2 = Node(code="b")
        n2.success = True
        n2.is_buggy = False
        n2.metric = MetricValue(0.81, maximize=False)
        n2.score = 0.81
        journal.append(n2)

        best = journal.get_best_node()
        self.assertIsNotNone(best)
        self.assertEqual(best.node_id, n2.node_id)

    def test_debug_depth(self):
        root = Node(code="root")
        root.is_buggy = True

        d1 = Node(code="d1", parent=root, stage="debug")
        d1.is_buggy = True
        d2 = Node(code="d2", parent=d1, stage="debug")
        d2.is_buggy = True

        self.assertEqual(root.debug_depth, 0)
        self.assertEqual(d1.debug_depth, 1)
        self.assertEqual(d2.debug_depth, 2)

    def test_worst_metric_for_buggy(self):
        n = Node(code="x")
        n.metric = WorstMetricValue(maximize=False)
        self.assertTrue(n.metric.is_worst)


if __name__ == "__main__":
    unittest.main()
