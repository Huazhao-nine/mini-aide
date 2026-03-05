import os
import subprocess
import sys
import time
from dataclasses import dataclass

from config import WORKSPACE_DIR


@dataclass
class ExecutionResult:
    success: bool
    output: str          # merged stdout+stderr
    error: str           # short error string (optional)
    execution_time: float


class Interpreter:
    """Lightweight subprocess interpreter (simplified, AIDE-aligned)."""

    def __init__(self, workdir: str = WORKSPACE_DIR, timeout: int = 60):
        self.workspace_dir = os.path.abspath(workdir)
        self.timeout = int(timeout)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def run(self, code: str, filename: str = "solution.py") -> ExecutionResult:
        file_path = os.path.join(self.workspace_dir, filename)

        # 1) write code
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Failed to write {filename}: {e}",
                execution_time=0.0,
            )

        # 2) run
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, filename],
                cwd=self.workspace_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # IMPORTANT: merge stderr into stdout
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            out = proc.stdout or ""
            ok = (proc.returncode == 0)
            err = "" if ok else f"Non-zero exit code: {proc.returncode}"
            return ExecutionResult(success=ok, output=out, error=err, execution_time=elapsed)

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            out = (e.stdout or "") + "\n" + (e.stderr or "")
            return ExecutionResult(
                success=False,
                output=out.strip(),
                error=f"Timeout after {self.timeout}s",
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return ExecutionResult(
                success=False,
                output="",
                error=f"Interpreter crashed: {e}",
                execution_time=elapsed,
            )