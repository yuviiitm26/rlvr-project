"""
Fake LLM Testing Harness for RLVR Pipeline Validation.

Provides deterministic, scripted responses that simulate various
model behaviors. This is essential for:

  1. Testing the grader → Does it correctly parse pass/fail/timeout?
  2. Testing the MDP loop → Does multi-turn feedback work?
  3. Testing code extraction → Does it handle edge cases?
  4. Testing GRPO advantages → Do rewards compute correctly?
  5. Testing DAPO filtering → Are zero-variance batches detected?

Each "strategy" simulates a behavioral archetype that a real model
might exhibit during training:

  Strategy                  │ Simulates...
  ──────────────────────────┼──────────────────────────────────────
  immediate_correct         │ Model solves on first attempt
  syntax_error_then_fix     │ Model makes syntax error, self-corrects
  runtime_error_then_fix    │ Model makes logic error, self-corrects
  progressive_fix           │ Model improves gradually over turns
  always_wrong              │ Model never produces correct code
  timeout_bomb              │ Model produces infinite loops
  format_chaos              │ Model outputs badly formatted responses

Why a Fake LLM?
  - GPU time is expensive. Debugging on Kaggle T4s costs real money.
  - End-to-end tests must be fast (<1s) and deterministic.
  - We need to test EVERY code path: pass, fail, timeout, format edge cases.
  - Real model inference is slow and non-deterministic.
"""

from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# Response Banks
# ════════════════════════════════════════════════════════════════════
# Organized as: RESPONSE_BANKS[problem_id][strategy][turn_number]
# This 3-level hierarchy makes it easy to add new problems and strategies.

RESPONSE_BANKS: Dict[str, Dict[str, Dict[int, str]]] = {
    # ──────────────────────────────────────────────────────────────
    # add_two_numbers
    # ──────────────────────────────────────────────────────────────
    "add_two_numbers": {
        "immediate_correct": {
            1: (
                "<think>\n"
                "I need to write a function that adds two numbers.\n"
                "This is straightforward addition.\n"
                "</think>\n\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "```"
            ),
        },
        "syntax_error_then_fix": {
            1: (
                "<think>\nLet me write the add function.\n</think>\n\n"
                "```python\n"
                "def add(a, b)\n"  # Missing colon — SyntaxError
                "    return a + b\n"
                "```"
            ),
            2: (
                "<think>\n"
                "I see — I'm missing a colon after the function definition.\n"
                "Let me fix that.\n"
                "</think>\n\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "```"
            ),
        },
        "runtime_error_then_fix": {
            1: (
                "<think>\nI need to add two numbers.\n</think>\n\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a * b\n"  # Wrong operator — AssertionError
                "```"
            ),
            2: (
                "<think>\n"
                "The test failed because I used multiplication instead "
                "of addition. The assertion `add(2, 3) == 5` failed "
                "because 2 * 3 = 6, not 5. Let me fix the operator.\n"
                "</think>\n\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "```"
            ),
        },
        "always_wrong": {
            1: (
                "```python\n"
                "def add(a, b):\n"
                "    return a - b\n"
                "```"
            ),
            2: (
                "```python\n"
                "def add(a, b):\n"
                "    return a * b\n"
                "```"
            ),
            3: (
                "```python\n"
                "def add(a, b):\n"
                "    return a // b\n"
                "```"
            ),
            4: (
                "```python\n"
                "def add(a, b):\n"
                "    return abs(a) + abs(b)\n"  # Fails for negative numbers
                "```"
            ),
        },
        "timeout_bomb": {
            1: (
                "```python\n"
                "def add(a, b):\n"
                "    while True:\n"
                "        pass\n"
                "```"
            ),
        },
        "format_chaos": {
            # No code blocks, no think tags — tests extraction fallback
            1: (
                "Sure! Here's my solution:\n\n"
                "def add(a, b):\n"
                "    return a + b\n\n"
                "This should work for all cases."
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────
    # fibonacci
    # ──────────────────────────────────────────────────────────────
    "fibonacci": {
        "immediate_correct": {
            1: (
                "<think>\n"
                "Fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).\n"
                "I'll use iteration for O(n) time and O(1) space.\n"
                "</think>\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    if n <= 0:\n"
                "        return 0\n"
                "    if n == 1:\n"
                "        return 1\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```"
            ),
        },
        "runtime_error_then_fix": {
            1: (
                "<think>\nFibonacci with recursion.\n</think>\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    return fibonacci(n-1) + fibonacci(n-2)\n"
                "```"
                # Missing base case → RecursionError
            ),
            2: (
                "<think>\n"
                "I got a RecursionError because I forgot the base cases.\n"
                "I need to handle n=0 → 0 and n=1 → 1.\n"
                "Also, recursion is slow — let me use iteration.\n"
                "</think>\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    if n <= 0:\n"
                "        return 0\n"
                "    if n == 1:\n"
                "        return 1\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```"
            ),
        },
        "progressive_fix": {
            1: (
                "```python\n"
                "def fibonacci(n):\n"
                "    if n == 0:\n"
                "        return 0\n"
                "    if n == 1:\n"
                "        return 1\n"
                "    return fibonacci(n-1) + fibonacci(n-2)\n"
                "```"
                # Correct but potentially slow for large n
                # Should still pass the given tests though
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────
    # reverse_string
    # ──────────────────────────────────────────────────────────────
    "reverse_string": {
        "immediate_correct": {
            1: (
                "<think>\nReverse a string using slicing.\n</think>\n\n"
                "```python\n"
                "def reverse_string(s):\n"
                "    return s[::-1]\n"
                "```"
            ),
        },
        "syntax_error_then_fix": {
            1: (
                "```python\n"
                "def reverse_string(s):\n"
                "    return s[::1]\n"  # Wrong step — returns original
                "```"
            ),
            2: (
                "<think>\n"
                "s[::1] returns the string unchanged. I need s[::-1] to reverse.\n"
                "</think>\n\n"
                "```python\n"
                "def reverse_string(s):\n"
                "    return s[::-1]\n"
                "```"
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────
    # is_palindrome
    # ──────────────────────────────────────────────────────────────
    "is_palindrome": {
        "immediate_correct": {
            1: (
                "<think>\n"
                "I need to check if a string is a palindrome considering "
                "only alphanumeric chars and ignoring case.\n"
                "1. Filter to alphanumeric\n"
                "2. Convert to lowercase\n"
                "3. Compare with reverse\n"
                "</think>\n\n"
                "```python\n"
                "def is_palindrome(s):\n"
                "    filtered = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    return filtered == filtered[::-1]\n"
                "```"
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────
    # two_sum
    # ──────────────────────────────────────────────────────────────
    "two_sum": {
        "immediate_correct": {
            1: (
                "<think>\n"
                "Classic two-sum with hash map. O(n) time.\n"
                "For each number, check if (target - num) is in the map.\n"
                "Return sorted indices.\n"
                "</think>\n\n"
                "```python\n"
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return sorted([seen[complement], i])\n"
                "        seen[num] = i\n"
                "    return []\n"
                "```"
            ),
        },
        "runtime_error_then_fix": {
            1: (
                "```python\n"
                "def two_sum(nums, target):\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(len(nums)):\n"  # Bug: should start from i+1
                "            if nums[i] + nums[j] == target:\n"
                "                return sorted([i, j])\n"
                "    return []\n"
                "```"
            ),
            2: (
                "<think>\n"
                "The issue is that I'm checking i==j which uses the same "
                "element twice. I need j to start from i+1.\n"
                "</think>\n\n"
                "```python\n"
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return sorted([seen[complement], i])\n"
                "        seen[num] = i\n"
                "    return []\n"
                "```"
            ),
        },
    },
    # ──────────────────────────────────────────────────────────────
    # max_subarray_sum
    # ──────────────────────────────────────────────────────────────
    "max_subarray_sum": {
        "immediate_correct": {
            1: (
                "<think>\n"
                "Kadane's algorithm: track current_sum and max_sum.\n"
                "At each element, decide: extend the current subarray "
                "or start a new one.\n"
                "</think>\n\n"
                "```python\n"
                "def max_subarray_sum(nums):\n"
                "    max_sum = current_sum = nums[0]\n"
                "    for num in nums[1:]:\n"
                "        current_sum = max(num, current_sum + num)\n"
                "        max_sum = max(max_sum, current_sum)\n"
                "    return max_sum\n"
                "```"
            ),
        },
    },
}


# ════════════════════════════════════════════════════════════════════
# Generic Fallback Responses
# ════════════════════════════════════════════════════════════════════
# Used when a strategy is requested for a problem that doesn't have
# problem-specific responses.

GENERIC_RESPONSES: Dict[str, Dict[int, str]] = {
    "immediate_correct": {
        1: "```python\n# Generic correct solution placeholder\npass\n```",
    },
    "always_wrong": {
        1: "```python\nraise NotImplementedError('Attempt 1')\n```",
        2: "```python\nraise ValueError('Attempt 2')\n```",
        3: "```python\nraise RuntimeError('Attempt 3')\n```",
        4: "```python\nraise Exception('Attempt 4')\n```",
    },
    "timeout_bomb": {
        1: "```python\nimport time\ntime.sleep(100)\n```",
    },
    "format_chaos": {
        1: "I think the answer is 42. Let me try:\nresult = 42\nprint(result)",
    },
}


# ════════════════════════════════════════════════════════════════════
# Fake LLM Class
# ════════════════════════════════════════════════════════════════════

class FakeLLM:
    """
    Deterministic mock LLM for testing the RLVR pipeline.

    Produces scripted responses based on the current problem ID,
    strategy, and turn number. This enables systematic testing of
    all grader/MDP code paths without actual model inference.

    The FakeLLM satisfies the LLMInterface protocol (duck typing),
    so it's a drop-in replacement for a real model in the MDP loop.

    Usage:
        llm = FakeLLM(strategy="syntax_error_then_fix")
        response = llm.generate(messages)

    Testing Pattern:
        # Test all strategies for a problem
        for strategy in FakeLLM.VALID_STRATEGIES:
            llm = FakeLLM(strategy=strategy, problem_id="fibonacci")
            mdp = MultiTurnMDP(grader, llm)
            trajectory = mdp.run_episode(problem)
            # Assert expected behavior...
    """

    VALID_STRATEGIES = [
        "immediate_correct",
        "syntax_error_then_fix",
        "runtime_error_then_fix",
        "progressive_fix",
        "always_wrong",
        "timeout_bomb",
        "format_chaos",
    ]

    def __init__(
        self,
        strategy: str = "immediate_correct",
        problem_id: Optional[str] = None,
    ):
        """
        Args:
            strategy:   Behavioral strategy (see VALID_STRATEGIES).
            problem_id: If set, overrides auto-detection from messages.
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid: {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.problem_id = problem_id
        self._turn_counter = 0

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a scripted response based on strategy and turn count.

        The turn number is tracked by an internal counter that increments
        with each call. This simulates the model seeing progressively
        longer conversation histories.

        Args:
            messages: Conversation history (system + user + assistant messages).

        Returns:
            Scripted response string.
        """
        self._turn_counter += 1
        turn = self._turn_counter

        # Resolve problem ID
        problem_id = self.problem_id or self._detect_problem(messages)

        # Look up the scripted response
        return self._lookup_response(problem_id, turn)

    def reset(self) -> None:
        """Reset turn counter for a new episode."""
        self._turn_counter = 0

    # ────────────────────────────────────────────────────────────────
    # Internal: Problem Detection
    # ────────────────────────────────────────────────────────────────

    def _detect_problem(self, messages: List[Dict[str, str]]) -> str:
        """
        Heuristically detect the problem ID from conversation messages.

        This enables the FakeLLM to work without explicit problem_id
        when the MDP naturally includes the problem description in
        the conversation.
        """
        full_text = " ".join(
            m.get("content", "") for m in messages
        ).lower()

        # Order matters — check more specific patterns first
        if "fibonacci" in full_text or "fib(" in full_text:
            return "fibonacci"
        if "palindrome" in full_text:
            return "is_palindrome"
        if "two_sum" in full_text or "two numbers" in full_text:
            # Disambiguate: "add two numbers" vs "two_sum"
            if "indices" in full_text or "two_sum" in full_text:
                return "two_sum"
            return "add_two_numbers"
        if "add" in full_text and ("sum" in full_text or "numbers" in full_text):
            return "add_two_numbers"
        if "reverse" in full_text and "string" in full_text:
            return "reverse_string"
        if "subarray" in full_text or "kadane" in full_text:
            return "max_subarray_sum"
        if "flatten" in full_text and "nested" in full_text:
            return "flatten_nested_list"
        if "lru" in full_text or "cache" in full_text:
            return "lru_cache"
        if "parenthes" in full_text and "generate" in full_text:
            return "valid_parentheses_generate"

        return "unknown"

    # ────────────────────────────────────────────────────────────────
    # Internal: Response Lookup
    # ────────────────────────────────────────────────────────────────

    def _lookup_response(self, problem_id: str, turn: int) -> str:
        """
        Look up the scripted response for this problem/strategy/turn.

        Lookup priority:
          1. Problem-specific response bank
          2. Generic fallback responses
          3. Ultimate fallback (empty pass statement)

        When a turn number exceeds the scripted range, the last
        available response is repeated (models plateau, they don't
        suddenly produce new behaviors).
        """
        # Try problem-specific bank first
        if problem_id in RESPONSE_BANKS:
            problem_bank = RESPONSE_BANKS[problem_id]
            if self.strategy in problem_bank:
                strategy_bank = problem_bank[self.strategy]
                if turn in strategy_bank:
                    return strategy_bank[turn]
                # Repeat last scripted response for later turns
                if strategy_bank:
                    max_turn = max(strategy_bank.keys())
                    return strategy_bank[max_turn]

        # Fall back to generic responses
        if self.strategy in GENERIC_RESPONSES:
            generic_bank = GENERIC_RESPONSES[self.strategy]
            if turn in generic_bank:
                return generic_bank[turn]
            if generic_bank:
                max_turn = max(generic_bank.keys())
                return generic_bank[max_turn]

        # Ultimate fallback — should never reach here in practice
        return "```python\npass\n```"
