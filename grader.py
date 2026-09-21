"""
Subprocess Grading Engine for RLVR Training.

Executes AI-generated code in a quarantined sandbox and produces
structured execution results with formatted feedback for multi-turn
reinforcement learning.

Security model:
  - Code runs as `sandboxuser` (unprivileged, no shell, no home)
  - Execution confined to /tmp/ai_workspace
  - Hard timeout with process kill
  - Output truncation to prevent memory bombs
  - No shell=True (injection-safe)

Design Rationale:
  The grader is the *reward function* — the single source of truth for
  the RL loop. Its output must be:
    1. Deterministic (same code → same result, always)
    2. Informative (tracebacks cleaned for model consumption, not humans)
    3. Secure (untrusted code must never escape the sandbox)
    4. Fast (grading is on the critical path of every training step)
"""

import subprocess
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionResult:
    """
    Immutable result of a single code execution attempt.

    Fields:
        stdout:             Captured standard output (truncated).
        stderr:             Captured standard error (truncated).
        return_code:        Process exit code (0 = success, -1 = timeout).
        timed_out:          Whether the process was killed for exceeding timeout.
        elapsed_seconds:    Wall-clock execution time.
        passed:             Whether the code passed all assertions.
        reward:             Scalar reward signal (0.0 or 1.0).
        formatted_feedback: Clean, model-consumable feedback string.
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
    raw_code: str = ""

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
        }


# ════════════════════════════════════════════════════════════════════
# Subprocess Grader
# ════════════════════════════════════════════════════════════════════

class SubprocessGrader:
    """
    Executes Python code in a sandboxed subprocess and returns structured results.

    The grader writes the AI's code + test assertions to a temporary file,
    executes it as the sandbox user, captures output, and formats feedback
    suitable for feeding back into the language model.

    Architecture:
        ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
        │  AI Response  │ ──► │  SubprocessGrader│ ──► │ExecutionResult│
        │  (raw code)   │     │  (sandbox exec)  │     │  (feedback)   │
        └──────────────┘     └─────────────────┘     └──────────────┘

    Usage:
        grader = SubprocessGrader(use_sandbox=False)  # local dev
        result = grader.grade(
            code="def add(a, b): return a + b",
            test_code="assert add(2, 3) == 5"
        )
        print(result.passed)            # True
        print(result.reward)            # 1.0
        print(result.formatted_feedback) # "[PASS] All test assertions passed."
    """

    def __init__(
        self,
        sandbox_user: str = "sandboxuser",
        workspace_dir: str = "/tmp/ai_workspace",
        timeout_seconds: int = 10,
        max_output_bytes: int = 10_000,
        python_executable: str = "python3",
        use_sandbox: bool = True,
    ):
        """
        Args:
            sandbox_user:      Linux username for sandboxed execution.
            workspace_dir:     Directory for temp script files.
            timeout_seconds:   Hard kill timeout for code execution.
            max_output_bytes:  Max bytes to capture from stdout/stderr.
            python_executable: Path to Python interpreter.
            use_sandbox:       If False, run code as current user (for local dev).
        """
        self.sandbox_user = sandbox_user
        self.workspace_dir = Path(workspace_dir)
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable
        self.use_sandbox = use_sandbox

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def grade(self, code: str, test_code: str) -> ExecutionResult:
        """
        Execute `code` followed by `test_code` and return structured result.

        Pipeline:
          1. Concatenate solution code + test assertions
          2. Write to a unique temp file in the sandbox workspace
          3. Execute via subprocess (as sandboxuser if sandbox enabled)
          4. Capture stdout, stderr, return code
          5. Parse and clean tracebacks
          6. Format feedback for the model
          7. Clean up temp file

        Args:
            code:      The AI-generated solution code.
            test_code: Assertion-based test code (e.g., `assert add(2,3) == 5`).

        Returns:
            ExecutionResult with pass/fail, reward, and formatted feedback.
        """
        full_script = self._build_script(code, test_code)
        script_path = self._write_temp_script(full_script)

        try:
            result = self._execute(script_path, raw_code=code)
        finally:
            # Always clean up the temp file
            try:
                script_path.unlink()
            except OSError:
                pass

        return result

    # ────────────────────────────────────────────────────────────────
    # Internal: Script Construction
    # ────────────────────────────────────────────────────────────────

    def _build_script(self, code: str, test_code: str) -> str:
        """
        Combine user code and test assertions into a single executable script.

        The delimiter comments help with traceback parsing — we can identify
        whether an error originated in the solution or the test harness.
        """
        return (
            "# -*- coding: utf-8 -*-\n"
            "# === AI-GENERATED SOLUTION ===\n"
            f"{code.strip()}\n\n"
            "# === TEST ASSERTIONS ===\n"
            f"{test_code.strip()}\n"
        )

    def _write_temp_script(self, script: str) -> Path:
        """
        Write script to a uniquely-named temp file in the sandbox workspace.

        UUID-based naming prevents collisions during parallel grading
        (important for GRPO where G completions grade concurrently).
        """
        filename = f"submission_{uuid.uuid4().hex[:12]}.py"
        script_path = self.workspace_dir / filename
        script_path.write_text(script, encoding="utf-8")

        if self.use_sandbox:
            # Readable by sandbox user, but not writable
            os.chmod(script_path, 0o644)

        return script_path

    # ────────────────────────────────────────────────────────────────
    # Internal: Execution
    # ────────────────────────────────────────────────────────────────

    def _execute(self, script_path: Path, raw_code: str = "") -> ExecutionResult:
        """
        Run the script in a subprocess with timeout and output capture.

        Security notes:
          - shell=False: prevents shell injection
          - Explicit env: strips inherited env vars (API keys, paths, etc.)
          - sudo -u: drops privileges to sandboxuser
          - -u flag: unbuffered output for accurate capture
        """
        # Build the command
        if self.use_sandbox:
            cmd = [
                "sudo", "-u", self.sandbox_user,
                "--", self.python_executable, "-u", str(script_path),
            ]
        else:
            cmd = [self.python_executable, "-u", str(script_path)]

        # Locked-down environment — strip everything inherited
        env = {
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": str(self.workspace_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            # Disable hash randomization for deterministic error messages
            "PYTHONHASHSEED": "0",
        }

        start_time = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout_seconds,
                env=env,
                cwd=str(self.workspace_dir),
                # shell=False is default — explicit for security documentation
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

        # Determine pass/fail: exit code 0 AND no timeout
        passed = (return_code == 0) and (not timed_out)
        reward = 1.0 if passed else 0.0

        # Format feedback for the model
        feedback = self._format_feedback(
            passed=passed,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
        )

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            timed_out=timed_out,
            elapsed_seconds=elapsed,
            passed=passed,
            reward=reward,
            formatted_feedback=feedback,
            raw_code=raw_code,
        )

    # ────────────────────────────────────────────────────────────────
    # Internal: Output Processing
    # ────────────────────────────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        """
        Truncate output to prevent memory bombs.

        An adversarial/buggy solution could print gigabytes — we cap it
        to keep the training pipeline stable.
        """
        if len(text) > self.max_output_bytes:
            return text[: self.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"
        return text

    def _format_feedback(
        self,
        passed: bool,
        timed_out: bool,
        stdout: str,
        stderr: str,
        elapsed: float,
    ) -> str:
        """
        Format execution results into clean feedback for the language model.

        Design philosophy:
          - The model learns from *feedback structure*, not just reward signal.
          - Good feedback is: concise, actionable, free of system noise.
          - Bad feedback: raw dumps of sudo errors, sandbox paths, kernel messages.

        The formatted_feedback field is what gets appended to the conversation
        history in the multi-turn MDP. It's the model's "eyes" into execution.
        """
        if passed:
            return "[PASS] All test assertions passed."

        if timed_out:
            return (
                f"[TIMEOUT] Code execution exceeded {self.timeout_seconds}s limit.\n"
                "Your solution likely contains an infinite loop or is too slow.\n"
                "Review your loop conditions and algorithmic complexity."
            )

        # Parse and clean the traceback
        cleaned_tb = self._clean_traceback(stderr)

        parts = ["[FAIL] Test assertions failed."]

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

        What we strip:
          - Sandbox file paths → replaced with <solution>
          - sudo warnings/messages
          - Permission/auth noise

        What we preserve:
          - Exception type and message (the model needs this!)
          - Line numbers (relative to the solution)
          - The full traceback chain (for multi-exception scenarios)
        """
        if not stderr.strip():
            return ""

        lines = stderr.strip().split("\n")
        cleaned = []

        for line in lines:
            # Replace sandbox paths with generic <solution>
            line = re.sub(
                r'File ".*?submission_[a-f0-9]+\.py"',
                'File "<solution>"',
                line,
            )
            # Also handle non-UUID temp file paths
            line = re.sub(
                r'File "/tmp/ai_workspace/.*?\.py"',
                'File "<solution>"',
                line,
            )

            # Skip sudo/permission noise — these are infrastructure artifacts,
            # not actionable information for the model
            if any(
                skip in line.lower()
                for skip in [
                    "sudo:",
                    "permission denied",
                    "we trust you",
                    "password",
                    "not in the sudoers",
                    "incident will be reported",
                ]
            ):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)
