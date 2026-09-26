"""
POSIX Sandbox Grader for RLVR Training.

Uses low-level POSIX OS primitives (setuid, setgid, setrlimit) via
subprocess preexec_fn to enforce kernel-level security constraints on
untrusted AI-generated code.

Security Layers (Defense in Depth):
  Layer 1: preexec_fn → os.setuid/setgid (privilege drop)
  Layer 2: resource.setrlimit RLIMIT_AS (memory cap)
  Layer 3: resource.setrlimit RLIMIT_CPU (CPU time cap)
  Layer 4: resource.setrlimit RLIMIT_FSIZE (file write cap)
  Layer 5: subprocess.timeout (wall-clock kill)
  Layer 6: os.setgroups([]) (supplementary group strip)
  Layer 7: Minimal env dict (no API keys, no PATH leaks)
  Layer 8: Output truncation (stdout/stderr memory bomb prevention)

Why preexec_fn Instead of sudo:
  sudo requires /etc/sudoers configuration, introduces shell parsing risks,
  and adds ~50ms latency per execution. preexec_fn runs in the forked child
  process *before* exec(), meaning the kernel enforces limits before a single
  byte of untrusted code executes. Zero overhead. Zero configuration.

Compatibility:
  This module exports the same ExecutionResult dataclass and a
  PythonJailGrader that is API-compatible with SubprocessGrader.
  The MDP can use either interchangeably.
"""

import subprocess
import os
import sys
import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

# POSIX-only imports (Linux/macOS)
try:
    import pwd
    import resource
    POSIX_AVAILABLE = True
except ImportError:
    POSIX_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
# Data Structures (Same interface as grader.py)
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionResult:
    """
    Immutable result of a single sandboxed code execution attempt.

    Fields:
        stdout:             Captured standard output (truncated).
        stderr:             Captured standard error (truncated).
        return_code:        Process exit code (0 = success, -1 = timeout, -9 = OOM killed).
        timed_out:          Whether the process was killed for exceeding timeout.
        elapsed_seconds:    Wall-clock execution time.
        passed:             Whether the code passed all assertions.
        reward:             Scalar reward signal (0.0 or 1.0).
        formatted_feedback: Clean, model-consumable feedback string.
        error_type:         Categorized error for MDP state tracking.
        raw_code:           The code that was executed (for logging).
    """
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    elapsed_seconds: float
    passed: bool
    reward: float
    formatted_feedback: str
    error_type: str = "None"
    raw_code: str = ""
    error_line: int = -1
    total_lines: int = 0

    def to_dict(self) -> dict:
        """Serialize to dict for JSON logging / trajectory storage."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "passed": self.passed,
            "reward": self.reward,
            "formatted_feedback": self.formatted_feedback,
            "error_type": self.error_type,
        }


# ════════════════════════════════════════════════════════════════════
# PythonJailGrader
# ════════════════════════════════════════════════════════════════════

class PythonJailGrader:
    """
    Executes untrusted Python code with POSIX kernel-level sandboxing.

    Security model:
      1. Code is written to a temp file in the workspace.
      2. subprocess.Popen forks a child process.
      3. preexec_fn fires in the child BEFORE exec():
         - Drops root → sandboxuser (os.setuid / os.setgid)
         - Strips supplementary groups (os.setgroups([]))
         - Caps virtual memory (RLIMIT_AS)
         - Caps CPU time (RLIMIT_CPU)
         - Caps file write size (RLIMIT_FSIZE)
      4. The untrusted code runs inside these kernel constraints.
      5. subprocess.timeout kills the process if wall-clock exceeds limit.
      6. Output is captured, truncated, cleaned, and returned.

    Architecture:
      ┌─────────────┐    fork()    ┌──────────────┐    exec()    ┌──────────┐
      │ Parent Proc  │───────────►│ Child Process │────────────►│ Untrusted│
      │ (full privs) │            │ preexec_fn:   │             │  Code    │
      │              │            │  setuid()     │             │          │
      │              │            │  setrlimit()  │             │          │
      │              │            │  setgroups([])│             │          │
      └─────────────┘            └──────────────┘             └──────────┘
           │                                                        │
           │◄──────────── stdout/stderr pipe ──────────────────────┘
           │
           ▼
      ExecutionResult
    """

    def __init__(
        self,
        workspace: str = "/tmp/ai_workspace",
        user: str = "sandboxuser",
        timeout: float = 5.0,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 6,
        max_file_size_mb: int = 5,
        max_output_bytes: int = 10_000,
        use_sandbox: bool = True,
    ):
        """
        Args:
            workspace:        Directory for temporary script files.
            user:             Unprivileged Linux user to execute code as.
            timeout:          Wall-clock timeout (seconds).
            max_memory_mb:    RLIMIT_AS cap (virtual memory, MB).
            max_cpu_seconds:  RLIMIT_CPU cap (CPU time, seconds).
            max_file_size_mb: RLIMIT_FSIZE cap (max file write size, MB).
            max_output_bytes: Max bytes to capture from stdout/stderr.
            use_sandbox:      If False, skip privilege drop (for local dev on Windows/macOS).
        """
        self.workspace = Path(workspace)
        self.user = user
        self.timeout = timeout
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_cpu_seconds = max_cpu_seconds
        self.max_file_bytes = max_file_size_mb * 1024 * 1024
        self.max_output_bytes = max_output_bytes
        self.use_sandbox = use_sandbox

        # Pre-compile regexes for fast traceback cleaning
        self.sandbox_path_pattern = re.compile(r'File ".*?submission_[a-f0-9]+\.py"')
        self.workspace_pattern = re.compile(r'File "/tmp/ai_workspace/.*?\.py"')
        # Resolve the target user's UID/GID
        self.target_user = None
        if self.use_sandbox:
            if not POSIX_AVAILABLE:
                raise RuntimeError(
                    "POSIX sandbox requires Linux. Set use_sandbox=False for local dev."
                )
            try:
                self.target_user = pwd.getpwnam(self.user)
            except KeyError:
                raise RuntimeError(
                    f"Sandbox user '{self.user}' does not exist. "
                    f"Run: useradd -M -s /bin/false {self.user}"
                )

        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    # Core: preexec_fn (runs in forked child before exec)
    # ────────────────────────────────────────────────────────────────

    def _make_preexec_fn(self):
        """
        Returns a closure that enforces kernel-level constraints.

        This function executes AFTER fork() but BEFORE exec().
        The untrusted code never runs without these limits active.
        """
        target_uid = self.target_user.pw_uid
        target_gid = self.target_user.pw_gid
        max_mem = self.max_memory_bytes
        max_cpu = self.max_cpu_seconds
        max_fsize = self.max_file_bytes

        def _set_limits():
            # Layer 1: Drop all supplementary groups
            os.setgroups([])

            # Layer 2: Switch to unprivileged group
            os.setgid(target_gid)

            # Layer 3: Switch to unprivileged user (MUST be after setgid!)
            os.setuid(target_uid)

            # Layer 4: Cap virtual memory (prevents OOM bombs)
            resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

            # Layer 5: Cap CPU time (backup to wall-clock timeout)
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu))

            # Layer 6: Cap file write size (prevents disk exhaustion)
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize, max_fsize))

            # Layer 7: Prevent fork bombs by disallowing new processes
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

        return _set_limits

    # ────────────────────────────────────────────────────────────────
    # Public API (compatible with SubprocessGrader)
    # ────────────────────────────────────────────────────────────────

    def grade(self, code: str, test_code: str) -> ExecutionResult:
        """
        Execute `code` + `test_code` in the sandbox and return structured result.

        This is the main interface consumed by the MDP loop.
        API-compatible with SubprocessGrader.grade().
        """
        full_script = self._build_script(code, test_code)
        script_path = self._write_temp_script(full_script)

        try:
            result = self._execute(script_path, raw_code=code)
        finally:
            # Always clean up to prevent inode exhaustion during GRPO rollouts
            try:
                script_path.unlink()
            except OSError:
                pass

        return result

    # ────────────────────────────────────────────────────────────────
    # Internal: Script Construction
    # ────────────────────────────────────────────────────────────────

    def _build_script(self, code: str, test_code: str) -> str:
        """Combine solution code and test assertions into a single executable."""
        return (
            "# -*- coding: utf-8 -*-\n"
            "# === AI-GENERATED SOLUTION ===\n"
            f"{code.strip()}\n\n"
            "# === TEST ASSERTIONS ===\n"
            f"{test_code.strip()}\n"
        )

    def _write_temp_script(self, script: str) -> Path:
        """Write script to a uniquely-named temp file in the workspace."""
        filename = f"submission_{uuid.uuid4().hex[:12]}.py"
        script_path = self.workspace / filename
        script_path.write_text(script, encoding="utf-8")

        # Make readable by sandboxuser, but not writable
        if self.use_sandbox:
            os.chmod(script_path, 0o644)

        return script_path

    # ────────────────────────────────────────────────────────────────
    # Internal: Sandboxed Execution
    # ────────────────────────────────────────────────────────────────

    def _execute(self, script_path: Path, raw_code: str = "") -> ExecutionResult:
        """
        Run the script in a sandboxed subprocess with preexec_fn enforcement.
        """
        cmd = [sys.executable, "-u", str(script_path)]

        # Minimal, locked-down environment
        env = {
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": str(self.workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }

        # Build preexec_fn (only on Linux with sandbox enabled)
        preexec = self._make_preexec_fn() if self.use_sandbox else None

        start_time = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                cmd,
                preexec_fn=preexec,
                capture_output=True,
                timeout=self.timeout,
                env=env,
                cwd=str(self.workspace),
            )
            elapsed = time.monotonic() - start_time
            stdout = self._truncate(proc.stdout.decode("utf-8", errors="replace"))
            stderr = self._truncate(proc.stderr.decode("utf-8", errors="replace"))
            return_code = proc.returncode

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start_time
            timed_out = True
            stdout = ""
            stderr = ""
            return_code = -1

        # Determine pass/fail
        passed = (return_code == 0) and (not timed_out)
        reward = 1.0 if passed else 0.0
        error_type = self._classify_error(return_code, stderr, timed_out)

        # Format feedback for the model
        feedback = self._format_feedback(
            passed=passed,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            error_type=error_type,
        )

        # Process Rewards (Execution Semantics Alignment)
        total_lines = len(raw_code.strip().split("\n"))
        error_line = -1
        
        if not passed and stderr:
            # Look for the last line number in the traceback related to the temp script
            matches = re.findall(r'File ".*?submission_[a-f0-9]+\.py", line (\d+)', stderr)
            if matches:
                # The traceback puts the deepest call last, but sometimes the first match is the main script execution.
                # If there are multiple, the deepest one in the user's code is usually the best indicator of failure line.
                # However, if it's an AssertionError in the tests, it will be the test line.
                # For safety, we take the *first* match that is within the raw_code boundary, or just the last match overall.
                # The header is 2 lines:
                # 1: # -*- coding: utf-8 -*-
                # 2: # === AI-GENERATED SOLUTION ===
                # 3: def ...
                header_offset = 2
                for m in reversed(matches):
                    lineno = int(m) - header_offset
                    if 1 <= lineno <= total_lines:
                        error_line = lineno
                        break
                
                # If we didn't find one in the AI code, it might be in the test code
                if error_line == -1:
                    error_line = int(matches[-1]) - header_offset

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            timed_out=timed_out,
            elapsed_seconds=elapsed,
            passed=passed,
            reward=reward,
            formatted_feedback=feedback,
            error_type=error_type,
            raw_code=raw_code,
            error_line=error_line,
            total_lines=total_lines,
        )

    # ────────────────────────────────────────────────────────────────
    # Internal: Error Classification
    # ────────────────────────────────────────────────────────────────

    def _classify_error(self, return_code: int, stderr: str, timed_out: bool) -> str:
        """
        Categorize the failure mode for MDP state tracking.

        These categories help the model understand *what kind* of error
        occurred, enabling it to select different repair strategies:
          - Syntax_Error → rewrite the code structure
          - Assertion_Failure → fix the logic
          - OOM_Exception → reduce memory usage
          - Timeout → fix infinite loops
          - Runtime_Crash → general debugging
        """
        if return_code == 0:
            return "None"
        if timed_out:
            return "Timeout"
        if "MemoryError" in stderr:
            return "OOM_Exception"
        if "MemoryError" in stderr or return_code == -9:
            return "OOM_Killed"
        if "AssertionError" in stderr:
            return "Assertion_Failure"
        if "SyntaxError" in stderr:
            return "Syntax_Error"
        if "NameError" in stderr:
            return "Name_Error"
        if "TypeError" in stderr:
            return "Type_Error"
        if "IndexError" in stderr:
            return "Index_Error"
        if "IOError" in stderr or "PermissionError" in stderr:
            return "Permission_Denied"
        return "Runtime_Crash"

    # ────────────────────────────────────────────────────────────────
    # Internal: Output Processing
    # ────────────────────────────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        """Truncate output to prevent memory bombs during GRPO rollouts."""
        if len(text) > self.max_output_bytes:
            return text[:self.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"
        return text

    def _format_feedback(
        self,
        passed: bool,
        timed_out: bool,
        stdout: str,
        stderr: str,
        elapsed: float,
        error_type: str,
    ) -> str:
        """
        Format execution results into clean, actionable feedback for the LLM.

        The feedback is appended to the conversation history in the multi-turn
        MDP. The model learns to interpret and act on this structured output.
        """
        if passed:
            return "[PASS] All test assertions passed."

        if timed_out:
            return (
                f"[TIMEOUT] Code execution exceeded {self.timeout}s limit.\n"
                "Your solution likely contains an infinite loop or is too slow.\n"
                "Review your loop conditions and algorithmic complexity."
            )

        # Clean the traceback
        cleaned_tb = self._clean_traceback(stderr)

        parts = [f"[FAIL] {error_type}: Test assertions failed."]

        if stdout.strip():
            parts.append(f"\n--- Program Output ---\n{stdout.strip()}")

        if cleaned_tb:
            parts.append(f"\n--- Error Traceback ---\n{cleaned_tb}")

        parts.append(
            "\nAnalyze the error, identify the bug in your solution, "
            "and provide a corrected version."
        )

        return "\n".join(parts)

    def _clean_traceback(self, stderr: str) -> str:
        """
        Strip sandbox-specific paths and system noise from tracebacks.

        Preserves exception types, messages, and line numbers — the
        information the model actually needs to debug its code.
        """
        if not stderr.strip():
            return ""

        lines = stderr.strip().split("\n")
        cleaned = []

        # Prepare skip keywords once
        if not hasattr(self, "skip_keywords"):
            self.skip_keywords = {
                "sudo:",
                "permission denied",
                "we trust you",
                "password",
                "not in the sudoers",
                "incident will be reported",
            }
            
        for line in lines:
            # Replace sandbox paths with generic <solution>
            line = self.sandbox_path_pattern.sub('File "<solution>"', line)
            line = self.workspace_pattern.sub('File "<solution>"', line)

            # Skip system noise
            line_lower = line.lower()
            if any(skip in line_lower for skip in self.skip_keywords):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)
