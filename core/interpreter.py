"""
无状态代码执行器。

论文把每个候选脚本都看成一个可独立评估的 solution s，并通过 objective function h(s)
得到分数。这个文件实现的就是“运行脚本并收集结果”的底层 evaluator：
- 单独的子进程负责执行候选代码，隔离主进程；
- stdout/stderr、异常、耗时都会被结构化收集；
- 超时后会中断子进程，避免坏脚本卡死整个搜索。
"""

import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

from config import WORKSPACE_DIR

EOF_TOKEN = "<|EOF|>"


@dataclass
class ExecutionResult:
    # Structured result fields：更接近论文里的 evaluator 输出。
    term_out: List[str]
    exec_time: float
    exc_type: Optional[str]
    exc_info: Optional[Dict[str, Any]] = None
    exc_stack: Optional[List[Tuple[str, int, str, Optional[str]]]] = None

    # Compatibility fields：为了让上层调用更简单，提供 success/output/error 这些派生属性。
    @property
    def success(self) -> bool:
        return self.exc_type is None

    @property
    def output(self) -> str:
        return "".join(self.term_out or [])

    @property
    def error(self) -> str:
        if self.exc_type is None:
            return ""
        if self.exc_info and self.exc_info.get("args"):
            args = ", ".join(str(x) for x in self.exc_info.get("args", []))
            return f"{self.exc_type}: {args}" if args else str(self.exc_type)
        return str(self.exc_type)

    @property
    def execution_time(self) -> float:
        return self.exec_time


class _QueueWriter:
    # 把子进程中的 stdout/stderr 重定向到 multiprocessing.Queue，主进程就能异步收集输出。
    def __init__(self, outq: Queue, timeout: float = 1.0):
        self.outq = outq
        self.timeout = timeout

    def write(self, msg: str) -> None:
        if not msg:
            return
        try:
            self.outq.put(msg, timeout=self.timeout)
        except Exception:
            return

    def flush(self) -> None:
        return


class Interpreter:
    """
    基于子进程的解释器。

    这里采用“持久子进程 + 多轮执行”的方式，而不是每次 `subprocess.run` 新开解释器。
    好处是协议统一、超时中断更明确，也更容易把输出流式写回主进程。
    """

    def __init__(self, workdir: str = WORKSPACE_DIR, timeout: int = 60, exec_filename: str = "solution.py"):
        self.workspace_dir = os.path.abspath(workdir)
        self.timeout = int(timeout)
        self.exec_filename = exec_filename

        # 所有候选脚本都在同一个工作目录下执行，这样 submission.csv、best.py 等文件路径稳定。
        os.makedirs(self.workspace_dir, exist_ok=True)

        self.process: Optional[Process] = None
        self.code_inq: Optional[Queue] = None
        self.result_outq: Optional[Queue] = None
        self.event_outq: Optional[Queue] = None

    def _child_proc_setup(self, result_outq: Queue) -> None:
        # 子进程启动后切换到工作目录，并把标准输出/错误输出接管到队列。
        os.chdir(self.workspace_dir)
        if self.workspace_dir not in sys.path:
            sys.path.append(self.workspace_dir)
        sys.stdout = _QueueWriter(result_outq)  # type: ignore[assignment]
        sys.stderr = _QueueWriter(result_outq)  # type: ignore[assignment]

    def _exception_summary(self, exc: BaseException) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, int, str, Optional[str]]]]:
        # 把异常变成可序列化结构，后续既能展示 traceback，也能在 prompt 中作为 debug 上下文。
        tb_str = "".join(traceback.format_exception(exc))
        tb_str = tb_str.replace(os.path.join(self.workspace_dir, self.exec_filename), self.exec_filename)

        exc_info: Dict[str, Any] = {}
        if hasattr(exc, "args"):
            exc_info["args"] = [str(i) for i in getattr(exc, "args", [])]
        for attr in ("name", "msg", "obj"):
            if hasattr(exc, attr):
                exc_info[attr] = str(getattr(exc, attr))

        tb = traceback.extract_tb(exc.__traceback__)
        exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]
        return tb_str, exc.__class__.__name__, exc_info, exc_stack

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
        # 子进程主循环：不断接收代码、写到临时文件、`exec` 执行，再回传状态事件。
        self._child_proc_setup(result_outq)
        global_scope: Dict[str, Any] = {}

        while True:
            payload = code_inq.get()
            if payload is None:
                break
            code = str(payload)
            file_path = os.path.join(self.workspace_dir, self.exec_filename)
            os.chdir(self.workspace_dir)

            # 先把代码落成文件，再以该文件名编译执行。这样 traceback 中的文件名更稳定，
            # 也更接近用户实际会保存/查看的脚本。
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.exec_filename, "exec"), global_scope)
            except BaseException as exc:  # noqa: BLE001
                tb_str, exc_type, exc_info, exc_stack = self._exception_summary(exc)
                result_outq.put(tb_str)
                if exc_type == "KeyboardInterrupt":
                    exc_type = "TimeoutError"
                event_outq.put(("state:finished", exc_type, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))
            finally:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
                result_outq.put(EOF_TOKEN)

    def _drain_queue(self, q: Optional[Queue]) -> None:
        # 执行前清空残留消息，避免上一轮的事件污染当前轮状态机。
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def create_process(self) -> None:
        # 真正启动 evaluator 子进程。
        self.code_inq = Queue()
        self.result_outq = Queue()
        self.event_outq = Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self) -> None:
        # 回收子进程。这里依次尝试 terminate / kill / SIGKILL，确保不会留下僵尸进程。
        if self.process is None:
            return
        try:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.5)
            if self.process.is_alive():
                os.kill(self.process.pid, signal.SIGKILL)
        except Exception:
            pass
        finally:
            try:
                if self.process is not None:
                    self.process.close()
            except Exception:
                pass
            self.process = None
            self.code_inq = None
            self.result_outq = None
            self.event_outq = None

    def _soft_interrupt_child(self) -> None:
        # 首选 SIGINT，让脚本有机会像 Ctrl+C 一样优雅退出；失败时再 terminate。
        if self.process is None or not self.process.is_alive():
            return
        try:
            os.kill(self.process.pid, signal.SIGINT)
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass

    def _collect_output_until_eof(self, max_wait_sec: float = 8.0) -> List[str]:
        # 持续收集输出，直到收到 EOF 标记或超时。
        assert self.result_outq is not None
        out: List[str] = []
        deadline = time.time() + max_wait_sec
        seen_eof = False

        while time.time() < deadline:
            try:
                msg = self.result_outq.get(timeout=0.2)
            except queue.Empty:
                if seen_eof:
                    break
                continue

            if msg == EOF_TOKEN:
                seen_eof = True
                break
            out.append(str(msg))
        return out

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """
        执行一份候选代码并返回结构化结果。

        这是上层眼中的 evaluator h(s) 入口，不过它只负责“运行并收集原始信号”；
        真正的 metric 提取和合规检查在 `Agent` 的 review 阶段完成。
        """
        if reset_session or self.process is None or not self.process.is_alive():
            self.cleanup_session()
            self.create_process()

        assert self.process is not None
        assert self.code_inq is not None and self.result_outq is not None and self.event_outq is not None

        self._drain_queue(self.result_outq)
        self._drain_queue(self.event_outq)

        self.code_inq.put(code)

        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            # 连 `state:ready` 都没收到，说明解释器协议本身出问题了。
            self.cleanup_session()
            return ExecutionResult(
                term_out=["Interpreter child failed to start execution.\n"],
                exec_time=0.0,
                exc_type="RuntimeError",
                exc_info={"args": ["child start timeout"]},
                exc_stack=[],
            )

        if not state or state[0] != "state:ready":
            self.cleanup_session()
            return ExecutionResult(
                term_out=["Interpreter protocol error: missing state:ready.\n"],
                exec_time=0.0,
                exc_type="RuntimeError",
                exc_info={"args": [str(state)]},
                exc_stack=[],
            )

        start = time.time()
        interrupted = False
        timeout_type: Optional[str] = None
        timeout_info: Optional[Dict[str, Any]] = None
        timeout_stack: Optional[List[Tuple[str, int, str, Optional[str]]]] = None

        while True:
            try:
                state = self.event_outq.get(timeout=0.2)
                if state and state[0] == "state:finished":
                    exec_time = time.time() - start
                    out = self._collect_output_until_eof()
                    return ExecutionResult(
                        term_out=out,
                        exec_time=exec_time,
                        exc_type=state[1],
                        exc_info=state[2],
                        exc_stack=state[3],
                    )
            except queue.Empty:
                pass

            elapsed = time.time() - start

            if not interrupted and self.timeout > 0 and elapsed > self.timeout:
                # 到达时间上限后先发软中断，尽量让用户脚本自己清理资源。
                interrupted = True
                self._soft_interrupt_child()

            if interrupted and elapsed > self.timeout + 5:
                # 若软中断后仍未退出，则认定为超时失败。
                timeout_type = "TimeoutError"
                timeout_info = {"args": [f"Execution exceeded {self.timeout}s"]}
                timeout_stack = []
                out = self._collect_output_until_eof(max_wait_sec=0.5) if self.result_outq else []
                self.cleanup_session()
                out.append(f"TimeoutError: Execution exceeded {self.timeout}s\n")
                return ExecutionResult(
                    term_out=out,
                    exec_time=float(self.timeout),
                    exc_type=timeout_type,
                    exc_info=timeout_info,
                    exc_stack=timeout_stack,
                )

            if self.process is None or not self.process.is_alive():
                out = self._collect_output_until_eof()
                if timeout_type is None:
                    timeout_type = "RuntimeError"
                    timeout_info = {"args": ["child process died unexpectedly"]}
                    timeout_stack = []
                return ExecutionResult(
                    term_out=out,
                    exec_time=time.time() - start,
                    exc_type=timeout_type,
                    exc_info=timeout_info,
                    exc_stack=timeout_stack,
                )
