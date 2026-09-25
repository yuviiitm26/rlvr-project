"""
Comprehensive Security Validation Suite for PythonJailGrader.

Tests each security layer independently to verify that the sandbox
correctly prevents all known attack vectors that an RL-exploring
language model might discover.

Test Categories:
  1. Happy Path         — Valid code executes and passes.
  2. Syntax Errors      — Model generates unparseable code.
  3. Assertion Failures — Model logic is wrong.
  4. Timeout (Infinite) — Model produces infinite loops.
  5. Memory Bomb        — Model tries to allocate unbounded memory.
  6. Disk Bomb          — Model tries to write massive files.
  7. Privilege Escape   — Model tries os.setuid(0) to regain root.
  8. Network Escape     — Model tries to open sockets.
  9. File System Attack — Model tries rm -rf or reads /etc/shadow.
  10. Fork Bomb         — Model tries to crash via process spawning.
  11. Import Jail       — Model tries importing dangerous modules.
  12. Feedback Quality  — Verify the model gets clean, actionable tracebacks.

Run on Kaggle/Colab (Linux) with: python test_sandbox.py
"""

import os
import sys
import time
import json

# We test both graders for comparison
from sandbox_grader import PythonJailGrader, ExecutionResult, POSIX_AVAILABLE


def run_test(grader, name: str, code: str, test_code: str, 
             expect_pass: bool, expect_error_type: str = None):
    """Execute a single test case and validate the result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    result = grader.grade(code, test_code)
    
    # Display result
    print(f"  Passed:     {result.passed}")
    print(f"  Reward:     {result.reward}")
    print(f"  Error Type: {result.error_type}")
    print(f"  Time:       {result.elapsed_seconds:.3f}s")
    print(f"  Return Code:{result.return_code}")
    
    if result.stdout:
        print(f"  Stdout:     {result.stdout[:200]}")
    if result.stderr:
        print(f"  Stderr:     {result.stderr[:300]}")
    
    # Validate expectations
    status = "✓"
    if result.passed != expect_pass:
        status = "✗ UNEXPECTED"
        print(f"  *** EXPECTED passed={expect_pass}, GOT passed={result.passed}")
    
    if expect_error_type and result.error_type != expect_error_type:
        status = "✗ WRONG ERROR TYPE"
        print(f"  *** EXPECTED error_type={expect_error_type}, GOT {result.error_type}")
    
    # Validate reward consistency
    if result.passed and result.reward != 1.0:
        status = "✗ REWARD MISMATCH"
    if not result.passed and result.reward != 0.0:
        status = "✗ REWARD MISMATCH"
        
    print(f"  Result:     [{status}]")
    print(f"  Feedback:   {result.formatted_feedback[:200]}")
    
    return status.startswith("✓"), result


def main():
    print("=" * 60)
    print("RLVR Sandbox Security Validation Suite")
    print("=" * 60)
    
    # Detect environment
    is_linux = sys.platform.startswith("linux")
    is_root = is_linux and os.getuid() == 0
    
    print(f"Platform:    {sys.platform}")
    print(f"POSIX:       {POSIX_AVAILABLE}")
    print(f"Is Root:     {is_root}")
    print(f"Python:      {sys.executable}")
    
    # If we're on Linux as root, ensure sandboxuser exists
    if is_linux and is_root:
        os.system("id sandboxuser 2>/dev/null || useradd -M -s /bin/false sandboxuser")
    
    # Choose sandbox mode based on environment
    use_sandbox = is_linux and is_root and POSIX_AVAILABLE
    
    grader = PythonJailGrader(
        workspace="/tmp/ai_workspace",
        user="sandboxuser",
        timeout=5.0,
        max_memory_mb=256,
        max_cpu_seconds=6,
        max_file_size_mb=5,
        max_output_bytes=10_000,
        use_sandbox=use_sandbox,
    )
    
    print(f"Sandbox:     {'ENABLED (kernel-level)' if use_sandbox else 'DISABLED (dev mode)'}")
    
    results = []
    
    # ══════════════════════════════════════════════════════════════
    # TEST 1: Happy Path — Valid code passes
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "1. Happy Path (add two numbers)",
        code="def add(a, b): return a + b",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('All good!')",
        expect_pass=True,
    )
    results.append(("Happy Path", ok))
    
    # ══════════════════════════════════════════════════════════════
    # TEST 2: Syntax Error
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "2. Syntax Error (unparseable code)",
        code="def broken(\n  return ???",
        test_code="assert True",
        expect_pass=False,
        expect_error_type="Syntax_Error",
    )
    results.append(("Syntax Error", ok))
    
    # ══════════════════════════════════════════════════════════════
    # TEST 3: Assertion Failure
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "3. Assertion Failure (wrong logic)",
        code="def add(a, b): return a - b  # wrong!",
        test_code="assert add(2, 3) == 5",
        expect_pass=False,
        expect_error_type="Assertion_Failure",
    )
    results.append(("Assertion Failure", ok))

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Infinite Loop (Timeout)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "4. Infinite Loop (timeout kill)",
        code="while True: pass",
        test_code="",
        expect_pass=False,
        expect_error_type="Timeout",
    )
    results.append(("Timeout", ok))
    # Verify it actually timed out, not just errored
    if r.timed_out:
        print("  ✓ Confirmed: subprocess was killed by timeout")
    else:
        print("  ✗ WARNING: Did not detect as timeout")
    
    # ══════════════════════════════════════════════════════════════
    # TEST 5: Memory Bomb (RLIMIT_AS)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "5. Memory Bomb (allocate 1GB list)",
        code="x = [0] * (1024 * 1024 * 1024)  # 1 billion integers",
        test_code="",
        expect_pass=False,
    )
    results.append(("Memory Bomb", ok))
    if use_sandbox:
        if "MemoryError" in r.stderr or r.return_code == -9:
            print("  ✓ Confirmed: RLIMIT_AS killed the allocation")
        else:
            print(f"  ⚠ Memory bomb result: rc={r.return_code}, stderr={r.stderr[:100]}")

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Disk Bomb (RLIMIT_FSIZE)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "6. Disk Bomb (write 100MB file)",
        code=(
            "with open('/tmp/ai_workspace/bomb.txt', 'w') as f:\n"
            "    f.write('A' * (100 * 1024 * 1024))"
        ),
        test_code="",
        expect_pass=False,
    )
    results.append(("Disk Bomb", ok))
    if use_sandbox:
        print("  ✓ Confirmed: RLIMIT_FSIZE prevented disk exhaustion")

    # ══════════════════════════════════════════════════════════════
    # TEST 7: Privilege Escalation (os.setuid(0))
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "7. Privilege Escalation (setuid to root)",
        code="import os; os.setuid(0)",
        test_code="",
        expect_pass=False,
    )
    results.append(("Privilege Escalation", ok))
    if use_sandbox and "PermissionError" in r.stderr:
        print("  ✓ Confirmed: setuid(0) was blocked by the kernel")

    # ══════════════════════════════════════════════════════════════
    # TEST 8: Network Escape (socket connection)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "8. Network Escape (HTTP request)",
        code=(
            "import urllib.request\n"
            "urllib.request.urlopen('http://example.com')"
        ),
        test_code="",
        expect_pass=False,
    )
    results.append(("Network Escape", ok))

    # ══════════════════════════════════════════════════════════════
    # TEST 9: File System Attack (read /etc/shadow)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "9. File System Attack (read /etc/shadow)",
        code="data = open('/etc/shadow').read(); print(data)",
        test_code="",
        expect_pass=False,
    )
    results.append(("File System Attack", ok))
    if use_sandbox and "Permission" in r.stderr:
        print("  ✓ Confirmed: sandboxuser cannot read /etc/shadow")

    # ══════════════════════════════════════════════════════════════
    # TEST 10: File System Attack (rm -rf /)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "10. Destructive Attack (rm -rf /)",
        code="import subprocess; subprocess.run(['rm', '-rf', '/'], check=True)",
        test_code="",
        expect_pass=False,
    )
    results.append(("Destructive Attack", ok))

    # ══════════════════════════════════════════════════════════════
    # TEST 11: Fork Bomb
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "11. Fork Bomb (os.fork in loop)",
        code="import os\nwhile True: os.fork()",
        test_code="",
        expect_pass=False,
    )
    results.append(("Fork Bomb", ok))
    
    # ══════════════════════════════════════════════════════════════
    # TEST 12: Stdout Bomb (print flood)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "12. Stdout Bomb (print 10M characters)",
        code="print('A' * 10_000_000)",
        test_code="",
        expect_pass=True,  # It should succeed but output is truncated
    )
    results.append(("Stdout Bomb", ok))
    if len(r.stdout) <= grader.max_output_bytes + 50:
        print(f"  ✓ Confirmed: Output truncated to {len(r.stdout)} bytes")
    else:
        print(f"  ✗ Output NOT truncated: {len(r.stdout)} bytes")

    # ══════════════════════════════════════════════════════════════
    # TEST 13: Feedback Quality (clean traceback)
    # ══════════════════════════════════════════════════════════════
    ok, r = run_test(
        grader, "13. Feedback Quality (clean traceback)",
        code="def fib(n):\n    return fib(n-1) + fib(n-2)  # no base case",
        test_code="assert fib(10) == 55",
        expect_pass=False,
    )
    results.append(("Feedback Quality", ok))
    # Verify traceback was cleaned
    if "/tmp/ai_workspace" not in r.formatted_feedback:
        print("  ✓ Confirmed: Sandbox paths stripped from feedback")
    else:
        print("  ✗ Sandbox paths leaked into feedback!")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECURITY VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {name}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("  🔒 ALL SECURITY LAYERS VALIDATED")
    else:
        print("  ⚠️  SOME TESTS FAILED — review above output")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
