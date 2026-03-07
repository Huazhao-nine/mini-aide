import tempfile
import unittest

from core.interpreter import Interpreter


class InterpreterTest(unittest.TestCase):
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
