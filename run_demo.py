"""
End-to-end demo of the RLVR pipeline.

Runs the Fake LLM through the Multi-Turn MDP with the Subprocess Grader
to validate all components work together correctly. This is the "smoke
test" that proves the entire pipeline is functional before deploying
to cloud GPUs.

Tests performed:
  1. Individual strategies on a single problem (grader correctness)
  2. GRPO group advantage computation (reward math)
  3. Multi-problem sweep (problem bank coverage)
  4. DAPO zero-variance batch detection (training efficiency)
  5. Format compliance reward (output structure)

Usage:
  # On cloud (Kaggle/Colab) with sandbox enabled:
  python3 run_demo.py

  # Local development (Windows/Mac/Linux) without sandbox:
  python3 run_demo.py --no-sandbox

  # Custom workspace and timeout:
  python3 run_demo.py --no-sandbox --workspace ./test_workspace --timeout 5
"""

import argparse
import json
import sys
import os
from typing import List

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grader import SubprocessGrader
from mdp import MultiTurnMDP, Trajectory
from fake_llm import FakeLLM
from problems import PROBLEMS, get_problem, get_problem_stats
from rewards import (
    RewardConfig,
    compute_grpo_advantages,
    should_skip_batch_dapo,
    format_compliance_reward,
)


# ════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════

def run_single_episode(
    grader: SubprocessGrader,
    problem: dict,
    strategy: str,
    max_turns: int = 4,
    reward_config: RewardConfig = None,
) -> Trajectory:
    """Run a single episode with a specific fake LLM strategy."""
    llm = FakeLLM(strategy=strategy, problem_id=problem["id"])
    mdp = MultiTurnMDP(
        grader=grader,
        llm=llm,
        max_turns=max_turns,
        reward_config=reward_config or RewardConfig(),
    )
    return mdp.run_episode(problem)


def run_grpo_group(
    grader: SubprocessGrader,
    problem: dict,
    strategies: List[str],
    max_turns: int = 4,
) -> dict:
    """
    Simulate a GRPO group: G completions of the same problem.

    Each 'strategy' plays the role of a different completion sampled
    from the policy π_θ. The advantages computed here are what would
    be used to update the policy gradient.
    """
    reward_config = RewardConfig(discount_gamma=0.9)
    trajectories = []

    for strategy in strategies:
        traj = run_single_episode(
            grader, problem, strategy, max_turns, reward_config
        )
        trajectories.append(traj)

    # Compute GRPO group advantages
    rewards = [t.final_reward for t in trajectories]
    advantages = compute_grpo_advantages(rewards)
    skip = should_skip_batch_dapo(rewards)

    return {
        "problem_id": problem["id"],
        "group_size": len(strategies),
        "rewards": [round(r, 4) for r in rewards],
        "advantages": [round(a, 4) for a in advantages],
        "dapo_skip": skip,
        "trajectories": [t.to_dict() for t in trajectories],
    }


# ════════════════════════════════════════════════════════════════════
# Test Suites
# ════════════════════════════════════════════════════════════════════

def test_individual_strategies(grader: SubprocessGrader) -> bool:
    """TEST 1: Run each strategy on add_two_numbers."""
    print("─" * 70)
    print("TEST 1: Individual Strategies on 'add_two_numbers'")
    print("─" * 70)

    problem = get_problem("add_two_numbers")
    all_passed = True

    strategies_expected = {
        "immediate_correct": {"solved": True, "max_turns": 1},
        "syntax_error_then_fix": {"solved": True, "max_turns": 2},
        "runtime_error_then_fix": {"solved": True, "max_turns": 2},
        "always_wrong": {"solved": False, "max_turns": 4},
        "timeout_bomb": {"solved": False, "max_turns": 4},
    }

    for strategy, expected in strategies_expected.items():
        traj = run_single_episode(grader, problem, strategy)
        result = traj.to_dict()

        # Print results
        status = "✓" if result["solved"] else "✗"
        print(f"\n  [{status}] Strategy: {strategy}")
        print(f"      Solved:  {result['solved']} (expected: {expected['solved']})")
        print(f"      Turns:   {result['num_turns']} (max expected: {expected['max_turns']})")
        print(f"      Reward:  {result['final_reward']}")

        for turn in result["turns"]:
            turn_status = "✓" if turn["passed"] else "✗"
            print(f"        Turn {turn['turn']}: [{turn_status}] "
                  f"reward={turn['reward']:.4f}")

        # Validate expectations
        if result["solved"] != expected["solved"]:
            print(f"      ⚠ UNEXPECTED: solved={result['solved']}")
            all_passed = False
        if result["num_turns"] > expected["max_turns"]:
            print(f"      ⚠ UNEXPECTED: too many turns")
            all_passed = False

    return all_passed


def test_grpo_advantages(grader: SubprocessGrader) -> bool:
    """TEST 2: GRPO group advantage computation."""
    print(f"\n{'─' * 70}")
    print("TEST 2: GRPO Group Advantage Computation")
    print("─" * 70)

    problem = get_problem("add_two_numbers")
    group = run_grpo_group(
        grader,
        problem,
        strategies=[
            "immediate_correct",       # Solves turn 1 → reward ≈ 1.0
            "runtime_error_then_fix",  # Solves turn 2 → reward ≈ 0.9
            "always_wrong",            # Never solves   → reward = 0.0
        ],
    )

    print(f"\n  Problem:    {group['problem_id']}")
    print(f"  Group size: {group['group_size']}")
    print(f"  Rewards:    {group['rewards']}")
    print(f"  Advantages: {group['advantages']}")
    print(f"  DAPO skip:  {group['dapo_skip']}")

    # Validate: advantages should sum to ~0 and first should be highest
    adv = group["advantages"]
    if abs(sum(adv)) > 0.01:
        print(f"  ⚠ Advantages don't sum to ~0: {sum(adv):.4f}")
        return False
    if not (adv[0] > adv[1] > adv[2]):
        print(f"  ⚠ Advantage ordering unexpected: first > second > third")
        return False

    print(f"  ✓ Advantages correctly normalized (sum ≈ 0, correct ordering)")
    return True


def test_multi_problem_sweep(grader: SubprocessGrader) -> bool:
    """TEST 3: Run immediate_correct on multiple problems."""
    print(f"\n{'─' * 70}")
    print("TEST 3: Multi-Problem Sweep (immediate_correct)")
    print("─" * 70)

    # Only test problems that have immediate_correct responses
    test_problems = ["add_two_numbers", "fibonacci", "reverse_string",
                     "is_palindrome", "two_sum", "max_subarray_sum"]

    all_passed = True
    for pid in test_problems:
        try:
            problem = get_problem(pid)
        except KeyError:
            print(f"  {pid:30s} [SKIP] Not in problem bank")
            continue

        traj = run_single_episode(grader, problem, "immediate_correct")
        status = "✓ SOLVED" if traj.solved else "✗ FAILED"
        print(
            f"  {pid:30s} [{status}] "
            f"turns={traj.num_turns} reward={traj.final_reward:.3f}"
        )

        if not traj.solved:
            all_passed = False

    return all_passed


def test_dapo_detection(grader: SubprocessGrader) -> bool:
    """TEST 4: DAPO zero-variance batch detection."""
    print(f"\n{'─' * 70}")
    print("TEST 4: DAPO Zero-Variance Batch Detection")
    print("─" * 70)

    problem = get_problem("add_two_numbers")
    all_passed = True

    # All pass → zero variance → SKIP
    all_pass = run_grpo_group(
        grader, problem,
        strategies=["immediate_correct", "immediate_correct",
                     "immediate_correct"],
    )
    skip_correct = all_pass["dapo_skip"] is True
    print(f"  All pass:  rewards={all_pass['rewards']}  "
          f"skip={all_pass['dapo_skip']}  "
          f"{'✓' if skip_correct else '⚠ WRONG'}")
    all_passed &= skip_correct

    # All fail → zero variance → SKIP
    all_fail = run_grpo_group(
        grader, problem,
        strategies=["always_wrong", "always_wrong", "always_wrong"],
    )
    skip_correct = all_fail["dapo_skip"] is True
    print(f"  All fail:  rewards={all_fail['rewards']}  "
          f"skip={all_fail['dapo_skip']}  "
          f"{'✓' if skip_correct else '⚠ WRONG'}")
    all_passed &= skip_correct

    # Mixed → nonzero variance → KEEP
    mixed = run_grpo_group(
        grader, problem,
        strategies=["immediate_correct", "always_wrong"],
    )
    skip_correct = mixed["dapo_skip"] is False
    print(f"  Mixed:     rewards={mixed['rewards']}  "
          f"skip={mixed['dapo_skip']}  "
          f"{'✓' if skip_correct else '⚠ WRONG'}")
    all_passed &= skip_correct

    return all_passed


def test_format_compliance() -> bool:
    """TEST 5: Format compliance reward computation."""
    print(f"\n{'─' * 70}")
    print("TEST 5: Format Compliance Reward")
    print("─" * 70)

    test_cases = [
        # (response, expected_reward, description)
        (
            "<think>\nReasoning here\n</think>\n\n```python\ncode\n```",
            1.0,
            "Perfect format (think + python block)",
        ),
        (
            "```python\ncode\n```",
            0.5,
            "Code block only (no think tags)",
        ),
        (
            "<think>\nJust thinking\n</think>\n\nNo code block here",
            0.5,
            "Think tags only (no code block)",
        ),
        (
            "Just plain text with no formatting at all",
            0.0,
            "No formatting",
        ),
    ]

    all_passed = True
    for response, expected, description in test_cases:
        actual = format_compliance_reward(response)
        match = abs(actual - expected) < 0.01
        status = "✓" if match else "⚠"
        print(f"  [{status}] {description:40s} → {actual:.1f} "
              f"(expected {expected:.1f})")
        all_passed &= match

    return all_passed


def test_problem_bank_integrity():
    """TEST 6: Verify problem bank is well-formed."""
    print(f"\n{'─' * 70}")
    print("TEST 6: Problem Bank Integrity")
    print("─" * 70)

    stats = get_problem_stats()
    print(f"  Total problems: {stats['total']}")
    print(f"  By difficulty:  {stats['by_difficulty']}")
    print(f"  By tag:         {stats['by_tag']}")

    all_passed = True
    for p in PROBLEMS:
        issues = []
        if not p.get("id"):
            issues.append("missing id")
        if not p.get("description"):
            issues.append("missing description")
        if not p.get("test_code"):
            issues.append("missing test_code")
        if not p.get("difficulty"):
            issues.append("missing difficulty")
        if "assert" not in p.get("test_code", ""):
            issues.append("test_code has no assertions")

        status = "✓" if not issues else "⚠"
        detail = f" ({', '.join(issues)})" if issues else ""
        print(f"  [{status}] {p['id']}{detail}")

        if issues:
            all_passed = False

    return all_passed


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RLVR Pipeline End-to-End Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_demo.py --no-sandbox           # Local development\n"
            "  python run_demo.py                        # Cloud with sandbox\n"
            "  python run_demo.py --no-sandbox --timeout 5\n"
        ),
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable sandbox (for local development on non-Linux systems)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory for code execution (default: OS temp dir)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Execution timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Default workspace based on platform
    if args.workspace is None:
        import tempfile
        args.workspace = os.path.join(tempfile.gettempdir(), "ai_workspace")

    print("=" * 70)
    print("  RLVR PIPELINE DEMO")
    print("  Subprocess Grader + Fake LLM + Multi-Turn MDP + GRPO Rewards")
    print("=" * 70)
    print(f"\n  Sandbox:   {'DISABLED (local dev)' if args.no_sandbox else 'ENABLED'}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Timeout:   {args.timeout}s")
    print(f"  Python:    {sys.executable}\n")

    # Initialize grader
    grader = SubprocessGrader(
        workspace_dir=args.workspace,
        timeout_seconds=args.timeout,
        use_sandbox=not args.no_sandbox,
        python_executable=sys.executable,  # Use the same Python that's running us
    )

    # Run all test suites
    results = {}

    # Tests that don't require subprocess execution
    results["format_compliance"] = test_format_compliance()
    results["problem_bank"] = test_problem_bank_integrity()

    # Tests that require subprocess execution (grader)
    results["individual_strategies"] = test_individual_strategies(grader)
    results["grpo_advantages"] = test_grpo_advantages(grader)
    results["multi_problem_sweep"] = test_multi_problem_sweep(grader)
    results["dapo_detection"] = test_dapo_detection(grader)

    # Summary
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {test_name}")
        all_passed &= passed

    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
