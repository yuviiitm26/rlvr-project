"""
Reward Functions for RLVR Training.

Implements deterministic, verifiable reward signals that serve as the
optimization target for the RL loop.

Key design constraint: ALL rewards must be deterministic and verifiable.
No learned reward models (that's RLHF). The compiler is the judge.

Reward Types:
  1. Binary:       R ∈ {0, 1} — simplest signal, pass or fail.
  2. Discounted:   γ^(t-1) — incentivizes fewer turns to solution.
  3. Format:       Bonus for structured output (code blocks, think tags).

Group Normalization (GRPO):
  A_i = (r_i - mean(R)) / (std(R) + ε)

  This is the core of GRPO: for each prompt, generate G completions,
  compute rewards, normalize within the group. The advantage tells the
  policy gradient which completions to reinforce (positive) vs. suppress
  (negative) *relative to the group*.

Algorithmic Improvements:
  - DAPO: Skip zero-variance groups (no gradient signal).
  - Dr. GRPO: No trajectory length normalization (prevents verbosity hacking).
"""

from dataclasses import dataclass
from typing import List

import numpy as np


# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════

@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    discount_gamma: float = 0.9
    """Per-turn discount factor. γ=0.9 means solving on turn 2 gives 0.9x reward."""

    binary_only: bool = False
    """If True, ignore discount — pure binary {0, 1} rewards."""

    format_reward_weight: float = 0.1
    """Weight for format compliance bonus in composite reward."""

    max_turns: int = 4
    """Maximum turns allowed in an episode."""


# ════════════════════════════════════════════════════════════════════
# Scalar Reward Functions
# ════════════════════════════════════════════════════════════════════

def binary_reward(passed: bool) -> float:
    """
    Simple binary reward: 1.0 for pass, 0.0 for fail.

    This is the baseline reward signal. It's maximally sparse —
    the model gets zero information about *how close* it was.
    But it's perfectly verifiable and deterministic.
    """
    return 1.0 if passed else 0.0


def discounted_reward(passed: bool, turn: int, gamma: float = 0.9) -> float:
    """
    Turn-discounted reward: γ^(t-1) for success on turn t.

    Incentivizes the model to solve problems in fewer attempts:
      Turn 1 correct → reward = 1.000  (γ^0)
      Turn 2 correct → reward = 0.900  (γ^1)
      Turn 3 correct → reward = 0.810  (γ^2)
      Turn 4 correct → reward = 0.729  (γ^3)
      Never correct  → reward = 0.000

    This creates a smooth gradient that rewards both correctness AND
    efficiency. The model learns that first-attempt solutions are more
    valuable than iterative debugging.

    Args:
        passed: Whether the code passed all tests.
        turn:   1-indexed turn number when the solution succeeded.
        gamma:  Discount factor ∈ (0, 1).

    Returns:
        Discounted reward ∈ [0, 1].
    """
    if not passed:
        return 0.0
    if turn < 1:
        raise ValueError(f"Turn must be >= 1, got {turn}")
    return gamma ** (turn - 1)


def dense_pass_ratio_reward(passed_tests: int, total_tests: int) -> float:
    """
    Dense reward signal based on fraction of passing tests.

    Instead of binary {0, 1}, returns k/N where k = number of passing
    assertions and N = total assertions. This helps the 0.5B model learn
    from partial successes during early training.

    Examples:
      5/5 tests pass → reward = 1.0  (full credit)
      3/5 tests pass → reward = 0.6  (partial credit)
      0/5 tests pass → reward = 0.0  (no credit)

    When to use: Early curriculum stages where the model frequently gets
    some assertions right but not all. The dense signal provides a
    smoother gradient than binary pass/fail.

    Args:
        passed_tests: Number of assertions that passed.
        total_tests:  Total number of assertions.

    Returns:
        Reward ∈ [0.0, 1.0].
    """
    if total_tests <= 0:
        return 0.0
    return min(passed_tests / total_tests, 1.0)


def format_compliance_reward(response: str) -> float:
    """
    Reward for proper response formatting.

    Why this matters: During RL training, the model can drift toward
    unstructured outputs. Format rewards provide gentle regularization
    that keeps outputs parseable by the code extraction pipeline.

    Checks:
      - Code wrapped in ```python ... ``` blocks  → +0.5
      - Reasoning wrapped in <think>...</think>    → +0.5

    Returns:
        Score ∈ [0.0, 1.0].
    """
    score = 0.0

    # Check for properly delimited code blocks
    if "```python" in response:
        # Verify the block is actually closed
        after_open = response[response.index("```python") + 10 :]
        if "```" in after_open:
            score += 0.5

    # Check for reasoning tags
    if "<think>" in response and "</think>" in response:
        # Verify think block appears before code (reasoning first)
        think_pos = response.index("<think>")
        code_pos = response.find("```python")
        if code_pos == -1 or think_pos < code_pos:
            score += 0.5

    return score


def composite_reward(
    exec_result: "ExecutionResult",
    turn: int,
    response: str,
    config: RewardConfig,
) -> float:
    """
    Weighted combination of execution reward, partial compilation reward, and format bonus.
    """
    passed = exec_result.passed
    if config.binary_only:
        exec_reward = binary_reward(passed)
    else:
        exec_reward = discounted_reward(passed, turn, config.discount_gamma)

    # Process Rewards (Min-Form Credit Assignment)
    # If the code threw a syntax error or runtime error, it failed at error_line.
    # We give a partial process reward based on how far it got!
    if not passed:
        error_line = getattr(exec_result, "error_line", -1)
        total_lines = getattr(exec_result, "total_lines", 0)
        
        if error_line > 0 and total_lines > 0:
            if error_line > total_lines:
                # The error happened inside the test code!
                # The model's logic compiled and ran successfully, but failed the assertion.
                exec_reward += 0.3
            else:
                # Code crashed mid-execution. Reward it for the lines it successfully navigated!
                progress = error_line / total_lines
                exec_reward += 0.1 + (0.2 * progress)
        elif getattr(exec_result, "error_type", "None") == "Assertion_Failure":
            exec_reward += 0.3

    fmt_reward = format_compliance_reward(response)

    return exec_reward + config.format_reward_weight * fmt_reward


# ════════════════════════════════════════════════════════════════════
# GRPO Group-Level Computation
# ════════════════════════════════════════════════════════════════════

def compute_grpo_advantages(
    rewards: List[float], codes: List[str] = None, epsilon: float = 1e-6
) -> List[float]:
    """
    Compute GRPO group-relative advantages.

        A_i = (r_i - mean(R)) / (std(R) + ε)

    If `codes` is provided, applies μ-GRPO (Inverse Frequency Scaling):
    Advantages are divided by the frequency of identical generated code strings
    to penalize mode collapse and frequency bias.
    """
    n = len(rewards)
    if n == 0:
        return []
        
    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = variance ** 0.5

    # Zero-variance guard (DAPO will skip these, but we handle gracefully)
    if std_r < epsilon:
        return [0.0] * n

    advantages = [(r - mean_r) / (std_r + epsilon) for r in rewards]

    # μ-GRPO Inverse Frequency Scaling
    if codes is not None and len(codes) == len(advantages):
        freqs = {}
        for c in codes:
            freqs[c] = freqs.get(c, 0) + 1
        
        for i, c in enumerate(codes):
            advantages[i] = advantages[i] / freqs[c]

    return advantages


def should_skip_batch_dapo(
    rewards: List[float], epsilon: float = 1e-6
) -> bool:
    """
    DAPO dynamic sampling: detect zero-variance reward batches.

    If all G completions for a prompt receive the same reward (all pass
    or all fail), the gradient signal is exactly zero — no completion
    is better or worse than any other. Training on these batches wastes
    compute and can introduce noise.

    DAPO prunes these batches from the training set, focusing compute
    on prompts that produce *informative* reward variance.

    Args:
        rewards:  List of rewards for G completions.
        epsilon:  Threshold for "effectively zero" variance.

    Returns:
        True if the batch should be skipped (zero variance).

    Example:
        >>> should_skip_batch_dapo([1.0, 1.0, 1.0])  # All pass
        True
        >>> should_skip_batch_dapo([0.0, 0.0, 0.0])  # All fail
        True
        >>> should_skip_batch_dapo([1.0, 0.0, 0.9])  # Mixed
        False
    """
    return float(np.std(rewards)) < epsilon
