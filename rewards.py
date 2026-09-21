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
    passed: bool,
    turn: int,
    response: str,
    config: RewardConfig,
) -> float:
    """
    Weighted combination of execution reward and format bonus.

    reward = execution_reward + weight * format_reward

    The format weight should be small (0.05-0.1) to avoid the model
    gaming format at the expense of correctness.
    """
    if config.binary_only:
        exec_reward = binary_reward(passed)
    else:
        exec_reward = discounted_reward(passed, turn, config.discount_gamma)

    fmt_reward = format_compliance_reward(response)

    return exec_reward + config.format_reward_weight * fmt_reward


# ════════════════════════════════════════════════════════════════════
# GRPO Group-Level Computation
# ════════════════════════════════════════════════════════════════════

def compute_grpo_advantages(
    rewards: List[float], epsilon: float = 1e-6
) -> List[float]:
    """
    Compute GRPO group-relative advantages.

        A_i = (r_i - mean(R)) / (std(R) + ε)

    For each prompt, we generate G completions and compute their rewards.
    The advantage normalizes rewards *within this group*, centering at zero.

    This means:
      - Completions better than the group average get positive advantage
      - Completions worse than average get negative advantage
      - The magnitude reflects how far from the mean they are

    The policy gradient then reinforces high-advantage completions
    and suppresses low-advantage ones.

    Dr. GRPO note: We do NOT divide by trajectory length here.
    Length normalization would incentivize the model to be verbose
    (more tokens = lower per-token loss = easier optimization).
    Stripping it forces the model to be concise.

    Args:
        rewards:  List of rewards for G completions of the same prompt.
        epsilon:  Small constant for numerical stability.

    Returns:
        List of normalized advantages (same length as rewards).

    Example:
        >>> compute_grpo_advantages([1.0, 0.9, 0.0])
        [0.7833, 0.5222, -1.3056]  # First-turn solve gets highest advantage
    """
    rewards_arr = np.array(rewards, dtype=np.float64)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std()

    # Zero-variance guard (DAPO will skip these, but we handle gracefully)
    if std_r < epsilon:
        return [0.0] * len(rewards)

    advantages = ((rewards_arr - mean_r) / (std_r + epsilon)).tolist()
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
