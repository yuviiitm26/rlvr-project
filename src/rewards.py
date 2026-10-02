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
import re
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



# ────────────────────────────────────────────────────────────────
# Process Reward Model (PRM) Scaffolding
# ────────────────────────────────────────────────────────────────

class HeuristicPRM:
    """
    Scaffolding for a Process Reward Model (PRM) ala OpenAI's o1.
    
    Currently implemented as a Heuristic PRM to evaluate the quality 
    of the intermediate reasoning steps inside <think> tags. 
    This class is designed to be hot-swappable with a Neural PRM (e.g., 
    a HuggingFace sequence classification model) when scaling up compute.
    """
    def __init__(self):
        # Keyword triggers for heuristic evaluation
        self.planning_words = [r"\bfirst\b", r"\bthen\b", r"\bfinally\b", r"\bstep \d+\b", r"^\s*\d+\.\s+"]
        self.edge_case_words = [r"edge case", r"empty", r"negative", r"zero", r"boundary", r"null", r"none"]
        self.reflection_words = [r"wait", r"actually", r"however", r"incorrect", r"let me", r"re-read", r"rethink", r"no,"]
        self.complexity_words = [r"time complexity", r"o\(n", r"space complexity", r"efficient", r"optimize"]

    def evaluate(self, think_content: str) -> float:
        """
        Evaluates the reasoning trace and assigns a dense process reward.
        Max process reward = 0.5
        """
        if not think_content or not think_content.strip():
            return 0.0

        think_lower = think_content.lower()
        prm_score = 0.0

        # 1. Structure (0.1): Did it break thoughts into multiple paragraphs?
        paragraphs = [p for p in think_content.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            prm_score += 0.1

        # 2. Planning (0.1): Did it formulate a step-by-step plan?
        if any(re.search(pattern, think_lower, re.MULTILINE) for pattern in self.planning_words):
            prm_score += 0.1

        # 3. Edge Cases (0.1): Did it consider boundary conditions?
        if any(re.search(pattern, think_lower) for pattern in self.edge_case_words):
            prm_score += 0.1

        # 4. Reflection (0.1): Did it exhibit self-correction or critique?
        if any(re.search(pattern, think_lower) for pattern in self.reflection_words):
            prm_score += 0.1

        # 5. Complexity (0.1): Did it analyze algorithm efficiency?
        if any(re.search(pattern, think_lower) for pattern in self.complexity_words):
            prm_score += 0.1

        return prm_score


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
    current_discount: float = None,
) -> float:
    """
    Implements a 4-tier Dense Reward Scorecard.
    Max Score = 2.0
    1. Format (+0.2 max)
    2. Syntax (+0.3 max)
    3. Logic (+1.0 max)
    4. Optimization (+0.5 max)
    """
    total_reward = 0.0

    # 1. Formatting Reward (+0.2 max)
    if "<think>" in response and "</think>" in response:
        total_reward += 0.1
    if "```python" in response and response.count("```") >= 2:
        total_reward += 0.1

    # 2. Syntax Reward (+0.3 max)
    # If the code compiled and ran far enough to hit an AssertionError (or it passed),
    # it means the Python syntax was valid.
    error_type = getattr(exec_result, "error_type", "None")
    syntax_failures = ["Syntax_Error", "Indentation_Error", "Name_Error", "Type_Error"]
    
    if error_type not in syntax_failures and not (error_type == "None" and not exec_result.passed and exec_result.return_code != 0):
        # We give the +0.3 if it passed, or if the failure was an assertion/timeout (meaning it compiled)
        total_reward += 0.3

    # 3. Logic Reward (+1.0 max)
    passed_tests = getattr(exec_result, "tests_passed", 0)
    total_tests = getattr(exec_result, "total_tests", 0)
    
    if total_tests > 0:
        logic_score = (passed_tests / total_tests) * 1.0
        # Discount logic score by turn to penalize multi-turn thrashing
        if current_discount is None:
            # Fallback to static discount
            discounted_logic = logic_score * (config.discount_gamma ** (turn - 1))
        else:
            # Use dynamic adaptive discount
            discounted_logic = logic_score * current_discount
        total_reward += discounted_logic

    # 4. Optimization Reward (+0.5 max)
    # Only award if the code actually solved the problem perfectly
    if exec_result.passed and passed_tests == total_tests and total_tests > 0:
        elapsed = getattr(exec_result, "elapsed_seconds", 999.0)
        if elapsed < 0.1:
            total_reward += 0.5
        elif elapsed < 0.5:
            # Partial credit for slightly slower code
            total_reward += 0.2

    # 5. Length & Efficiency Penalty (Max -0.5 penalty)
    # Discourages rambling in the <think> tags or generating bloated code.
    # We apply a -0.05 penalty for every 1000 characters generated.
    length_penalty = - (len(response) / 1000.0) * 0.05
    length_penalty = max(length_penalty, -0.5) # Cap the penalty at -0.5
    total_reward += length_penalty

    # 6. Process Reward Model (PRM) (+0.5 max)
    # Evaluates the internal logic steps independently of the outcome.
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)
        prm = HeuristicPRM()
        prm_score = prm.evaluate(think_content)
        total_reward += prm_score

    # 7. Self-Verification Reward (+0.5 max)
    # Rewards the model if it successfully wrote its own test cases AND passed the real tests.
    sv_passed = getattr(exec_result, "sv_passed", 0)
    sv_total = getattr(exec_result, "sv_total", 0)
    
    if exec_result.passed and passed_tests == total_tests and total_tests > 0:
        if sv_total > 0:
            # Massive bonus for self-verification
            if sv_passed == sv_total and sv_total >= 2:
                total_reward += 0.5
            else:
                # Partial bonus
                total_reward += (sv_passed / sv_total) * 0.2

    # 8. AST Complexity (Pythonic Code) Bonus/Penalty (-0.3 to +0.2)
    # Rewards flat, readable code and penalizes deeply nested spaghetti code.
    if exec_result.passed and passed_tests == total_tests and total_tests > 0:
        import ast
        try:
            tree = ast.parse(exec_result.raw_code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.ListComp, ast.DictComp)):
                    complexity += 1
            
            if complexity <= 3:
                total_reward += 0.2 # Very flat, elegant
            elif complexity <= 5:
                total_reward += 0.1 # Standard, clean
            elif complexity > 10:
                total_reward -= 0.3 # High cyclomatic complexity (Spaghetti)
            elif complexity > 7:
                total_reward -= 0.1 # Starting to get messy
        except Exception:
            pass # Ignore parsing errors here (handled by syntax reward)

    return total_reward


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
