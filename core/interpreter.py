import subprocess
import sys
import os
import time
from dataclasses import dataclass
from config import WORKSPACE_DIR
# 结果类，方便 Agent 解析
@dataclass
class ExecutionResult:
    success: bool       # 是否执行成功 (exit_code == 0)
    output: str         # stdout 和 stderr 的合并输出
    error: str          # 放报错信息（简化版）
    execution_time: float

class Interpreter:
    def __init__(self, workspace_dir: str = WORKSPACE_DIR, timeout: int = 60):
        """
        初始化代码执行环境
        :param workspace_dir: 代码保存和运行的工作目录
        :param timeout: 运行超时时间（秒）
        """
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.timeout = timeout        
        # 确保工作目录存在
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)
            print(f"📂 [Interpreter] Created workspace at: {self.workspace_dir}")
            
    def run(self, code: str, filename: str = "solution.py") -> ExecutionResult:
        """
        执行代码的核心逻辑：写文件 -> 子进程运行 -> 捕获输出
        """
        file_path = os.path.join(self.workspace_dir, filename)        
        # 1. 将 LLM 生成的代码写入文件
        # 先保存，再运行
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return ExecutionResult(False, "", f"Failed to write code to file: {str(e)}", 0.0)
        # 2. 准备运行命令
        # 使用 sys.executable 确保用的是当前环境的 Python (这样能用到你装好的 pytorch/sklearn)
        cmd = [sys.executable, filename]        
        start_time = time.time()        
        try:
            # 3. 调用 subprocess 运行
            # cwd=self.workspace_dir 保证代码在指定目录下运行，方便读写相对路径的数据集
            print(f"🏃 [Interpreter] Running {filename} in {self.workspace_dir} ...")            
            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True, # 捕获 stdout/stderr
                text=True,           # 以字符串形式返回，而不是 bytes
                timeout=self.timeout
            )            
            duration = time.time() - start_time            
            # 4. 封装结果
            # 将 stdout 和 stderr 合并，因为 Debug 时都需要看
            full_output = result.stdout + "\n" + result.stderr            
            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    output=full_output.strip(),
                    error="",
                    execution_time=duration
                )
            else:
                return ExecutionResult(
                    success=False,
                    output=result.stdout.strip(), 
                    error=result.stderr.strip(),  # 主要是 stderr 里的 Traceback
                    execution_time=duration
                )
        except subprocess.TimeoutExpired:
            # 处理超时情况
            return ExecutionResult(
                success=False,
                output="",
                error=f"TimeoutError: Code execution exceeded {self.timeout} seconds.",
                execution_time=self.timeout
            )
        except Exception as e:
            # 处理其他系统级错误
            return ExecutionResult(
                success=False,
                output="",
                error=f"SystemError: {str(e)}",
                execution_time=time.time() - start_time
            )