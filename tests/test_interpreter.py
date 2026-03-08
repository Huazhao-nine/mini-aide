"""
`core.interpreter` 的回归测试。

论文中的 evaluator 要求每个候选脚本都能被独立执行并得到结构化反馈。这里的测试负责保护
这个工程语义：
- 正常脚本会被识别为成功执行；
- 异常脚本会返回异常类型和栈信息；
- 超时脚本会被中断并标记为 TimeoutError。

复试时可以把它概括成：这些测试确保“h(s) 的执行壳”不会因为重构而失效。
"""

import tempfile
import unittest

from core.interpreter import Interpreter


class InterpreterTest(unittest.TestCase):
    """验证 evaluator 壳层的成功、异常和超时三类基础行为。"""

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mini_aide_interp_test_")
        self.interpreter = Interpreter(workdir=self.workdir, timeout=1)

    def tearDown(self):
        self.interpreter.cleanup_session()

    def test_run_success(self):
        try:
            result = self.interpreter.run("print('FINAL_MSE=0.88')\n")
        except PermissionError:
            self.skipTest("multiprocessing not permitted in this environment")
        self.assertTrue(result.success)
        self.assertIsNone(result.exc_type)

    def test_run_exception(self):
        try:
            result = self.interpreter.run("raise ValueError('x')\n")
        except PermissionError:
            self.skipTest("multiprocessing not permitted in this environment")
        self.assertFalse(result.success)
        self.assertEqual(result.exc_type, "ValueError")
        self.assertTrue(bool(result.exc_stack))

    def test_run_timeout(self):
        try:
            result = self.interpreter.run("import time\ntime.sleep(3)\n")
        except PermissionError:
            self.skipTest("multiprocessing not permitted in this environment")
        self.assertFalse(result.success)
        self.assertEqual(result.exc_type, "TimeoutError")


if __name__ == "__main__":
    unittest.main()
