"""
Problem Bank for RLVR Training and Testing.

Each problem is a self-contained unit with:
  - id:           Unique string identifier for logging and lookup.
  - description:  Human-readable problem statement (fed to the model).
  - test_code:    Python assertions the solution must pass (the reward signal).
  - starter_code: (optional) Code skeleton for guided problems.
  - difficulty:   easy | medium | hard — for curriculum learning.
  - tags:         Topic tags for stratified sampling.

Design Notes:
  - Test assertions are the *ground truth*. They must be:
    1. Deterministic (no randomness, no floating-point edge cases)
    2. Complete (cover edge cases the model might miss)
    3. Fast (execute in <1s — they run on every training step)

  - Problem descriptions should be clear but not over-specified.
    The model must learn to infer edge cases from test failures,
    not from exhaustive specs. This trains genuine reasoning.

  - Problems are ordered roughly by difficulty for curriculum learning:
    easy problems → stable initial gradients → harder problems later.
"""

from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# Problem Definitions
# ════════════════════════════════════════════════════════════════════

PROBLEMS: List[Dict] = [
    # ──────────────────────────────────────────────────────────────
    # EASY: Warm-up problems for initial training stability
    # ──────────────────────────────────────────────────────────────
    {
        "id": "add_two_numbers",
        "description": (
            "Write a function `add(a, b)` that returns the sum of two numbers.\n\n"
            "Examples:\n"
            "  add(2, 3) → 5\n"
            "  add(-1, 1) → 0\n"
            "  add(0, 0) → 0"
        ),
        "test_code": (
            "assert add(2, 3) == 5\n"
            "assert add(-1, 1) == 0\n"
            "assert add(0, 0) == 0\n"
            "assert add(100, -100) == 0\n"
            "assert add(1.5, 2.5) == 4.0\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["arithmetic", "warmup"],
    },
    {
        "id": "reverse_string",
        "description": (
            "Write a function `reverse_string(s)` that returns the reversed string.\n\n"
            "Examples:\n"
            "  reverse_string('hello') → 'olleh'\n"
            "  reverse_string('') → ''\n"
            "  reverse_string('a') → 'a'"
        ),
        "test_code": (
            "assert reverse_string('hello') == 'olleh'\n"
            "assert reverse_string('') == ''\n"
            "assert reverse_string('a') == 'a'\n"
            "assert reverse_string('racecar') == 'racecar'\n"
            "assert reverse_string('12345') == '54321'\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["strings"],
    },
    {
        "id": "fibonacci",
        "description": (
            "Write a function `fibonacci(n)` that returns the nth Fibonacci number.\n\n"
            "The Fibonacci sequence: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)\n\n"
            "Examples:\n"
            "  fibonacci(0) → 0\n"
            "  fibonacci(1) → 1\n"
            "  fibonacci(10) → 55"
        ),
        "test_code": (
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(2) == 1\n"
            "assert fibonacci(5) == 5\n"
            "assert fibonacci(10) == 55\n"
            "assert fibonacci(20) == 6765\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["recursion", "dynamic_programming"],
    },
    {
        "id": "is_palindrome",
        "description": (
            "Write a function `is_palindrome(s)` that checks if a string "
            "is a palindrome.\n"
            "Consider only alphanumeric characters and ignore case.\n\n"
            "Examples:\n"
            "  is_palindrome('A man, a plan, a canal: Panama') → True\n"
            "  is_palindrome('race a car') → False\n"
            "  is_palindrome('') → True"
        ),
        "test_code": (
            "assert is_palindrome('A man, a plan, a canal: Panama') == True\n"
            "assert is_palindrome('race a car') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome(' ') == True\n"
            "assert is_palindrome('ab') == False\n"
            "assert is_palindrome('aba') == True\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["strings", "two_pointers"],
    },
    {
        "id": "count_vowels",
        "description": (
            "Write a function `count_vowels(s)` that counts the number of vowels "
            "(a, e, i, o, u) in a string, case-insensitively.\n\n"
            "Examples:\n"
            "  count_vowels('hello') → 2\n"
            "  count_vowels('rhythm') → 0\n"
            "  count_vowels('AEIOU') → 5"
        ),
        "test_code": (
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('rhythm') == 0\n"
            "assert count_vowels('AEIOU') == 5\n"
            "assert count_vowels('') == 0\n"
            "assert count_vowels('aEiOu') == 5\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["strings", "counting"],
    },
    {
        "id": "max_of_three",
        "description": (
            "Write a function `max_of_three(a, b, c)` that returns the maximum "
            "of three numbers.\n\n"
            "Examples:\n"
            "  max_of_three(1, 2, 3) → 3\n"
            "  max_of_three(3, 2, 1) → 3\n"
            "  max_of_three(-1, -2, -3) → -1"
        ),
        "test_code": (
            "assert max_of_three(1, 2, 3) == 3\n"
            "assert max_of_three(3, 2, 1) == 3\n"
            "assert max_of_three(-1, -2, -3) == -1\n"
            "assert max_of_three(5, 5, 5) == 5\n"
            "assert max_of_three(0, -1, 1) == 1\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["arithmetic", "logic"],
    },
    {
        "id": "is_even",
        "description": (
            "Write a function `is_even(n)` that returns True if a number is even, "
            "and False otherwise.\n\n"
            "Examples:\n"
            "  is_even(2) → True\n"
            "  is_even(3) → False\n"
            "  is_even(0) → True"
        ),
        "test_code": (
            "assert is_even(2) == True\n"
            "assert is_even(3) == False\n"
            "assert is_even(0) == True\n"
            "assert is_even(-4) == True\n"
            "assert is_even(-7) == False\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["arithmetic", "logic"],
    },
    {
        "id": "sum_list",
        "description": (
            "Write a function `sum_list(nums)` that returns the sum of all elements "
            "in a list.\n\n"
            "Examples:\n"
            "  sum_list([1, 2, 3]) → 6\n"
            "  sum_list([]) → 0\n"
            "  sum_list([10]) → 10"
        ),
        "test_code": (
            "assert sum_list([1, 2, 3]) == 6\n"
            "assert sum_list([]) == 0\n"
            "assert sum_list([10]) == 10\n"
            "assert sum_list([-1, 1]) == 0\n"
            "assert sum_list([1.5, 2.5]) == 4.0\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["lists", "arithmetic"],
    },
    {
        "id": "remove_duplicates",
        "description": (
            "Write a function `remove_duplicates(lst)` that removes duplicate "
            "elements from a list while preserving their original order.\n\n"
            "Examples:\n"
            "  remove_duplicates([1, 2, 2, 3, 3, 3]) → [1, 2, 3]\n"
            "  remove_duplicates([]) → []\n"
            "  remove_duplicates([3, 1, 2, 1, 3]) → [3, 1, 2]"
        ),
        "test_code": (
            "assert remove_duplicates([1, 2, 2, 3, 3, 3]) == [1, 2, 3]\n"
            "assert remove_duplicates([]) == []\n"
            "assert remove_duplicates([1, 1, 1]) == [1]\n"
            "assert remove_duplicates([1, 2, 3]) == [1, 2, 3]\n"
            "assert remove_duplicates([3, 1, 2, 1, 3]) == [3, 1, 2]\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["lists"],
    },
    {
        "id": "count_words",
        "description": (
            "Write a function `count_words(sentence)` that returns the number of "
            "words in a sentence (words separated by whitespace).\n\n"
            "Examples:\n"
            "  count_words('hello world') → 2\n"
            "  count_words('') → 0\n"
            "  count_words('  spaces  ') → 1"
        ),
        "test_code": (
            "assert count_words('hello world') == 2\n"
            "assert count_words('') == 0\n"
            "assert count_words('one') == 1\n"
            "assert count_words('  spaces  ') == 1\n"
            "assert count_words('a b c d') == 4\n"
            "print('All tests passed!')"
        ),
        "difficulty": "easy",
        "tags": ["strings"],
    },
    # ──────────────────────────────────────────────────────────────
    # MEDIUM: Core algorithmic reasoning
    # ──────────────────────────────────────────────────────────────
    {
        "id": "two_sum",
        "description": (
            "Write a function `two_sum(nums, target)` that returns the indices "
            "of two numbers in `nums` that add up to `target`.\n\n"
            "You may assume each input has exactly one solution, and you may "
            "not use the same element twice. Return the indices as a sorted "
            "list [i, j] where i < j.\n\n"
            "Examples:\n"
            "  two_sum([2, 7, 11, 15], 9) → [0, 1]\n"
            "  two_sum([3, 2, 4], 6) → [1, 2]"
        ),
        "test_code": (
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
            "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
            "assert two_sum([3, 3], 6) == [0, 1]\n"
            "assert two_sum([1, 2, 3, 4, 5], 9) == [3, 4]\n"
            "print('All tests passed!')"
        ),
        "difficulty": "medium",
        "tags": ["hash_map", "arrays"],
    },
    {
        "id": "max_subarray_sum",
        "description": (
            "Write a function `max_subarray_sum(nums)` that returns the "
            "maximum sum of a contiguous subarray (Kadane's algorithm).\n\n"
            "The array will contain at least one element.\n\n"
            "Examples:\n"
            "  max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) → 6\n"
            "  max_subarray_sum([1]) → 1\n"
            "  max_subarray_sum([-1]) → -1"
        ),
        "test_code": (
            "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6\n"
            "assert max_subarray_sum([1]) == 1\n"
            "assert max_subarray_sum([-1]) == -1\n"
            "assert max_subarray_sum([5,4,-1,7,8]) == 23\n"
            "assert max_subarray_sum([-2, -1]) == -1\n"
            "print('All tests passed!')"
        ),
        "difficulty": "medium",
        "tags": ["dynamic_programming", "arrays"],
    },
    {
        "id": "flatten_nested_list",
        "description": (
            "Write a function `flatten(lst)` that takes a nested list of "
            "integers and returns a flat list of all integers.\n\n"
            "Examples:\n"
            "  flatten([1, [2, 3], [4, [5, 6]]]) → [1, 2, 3, 4, 5, 6]\n"
            "  flatten([]) → []\n"
            "  flatten([[[[1]]]]) → [1]"
        ),
        "test_code": (
            "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\n"
            "assert flatten([]) == []\n"
            "assert flatten([[[[1]]]]) == [1]\n"
            "assert flatten([1, 2, 3]) == [1, 2, 3]\n"
            "assert flatten([[], [1], [[], [2]]]) == [1, 2]\n"
            "print('All tests passed!')"
        ),
        "difficulty": "medium",
        "tags": ["recursion", "lists"],
    },
    # ──────────────────────────────────────────────────────────────
    # HARD: Multi-step reasoning required
    # ──────────────────────────────────────────────────────────────
    {
        "id": "lru_cache",
        "description": (
            "Implement an LRU (Least Recently Used) cache class.\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int): ...\n"
            "    def get(self, key: int) -> int: ...\n"
            "    def put(self, key: int, value: int) -> None: ...\n\n"
            "- `get(key)` returns the value if key exists, otherwise -1.\n"
            "- `put(key, value)` inserts or updates the key-value pair.\n"
            "  If the cache exceeds capacity, evict the least recently used key.\n"
            "- Both operations must be O(1) average time."
        ),
        "test_code": (
            "cache = LRUCache(2)\n"
            "cache.put(1, 1)\n"
            "cache.put(2, 2)\n"
            "assert cache.get(1) == 1\n"
            "cache.put(3, 3)  # Evicts key 2\n"
            "assert cache.get(2) == -1\n"
            "cache.put(4, 4)  # Evicts key 1\n"
            "assert cache.get(1) == -1\n"
            "assert cache.get(3) == 3\n"
            "assert cache.get(4) == 4\n"
            "print('All tests passed!')"
        ),
        "difficulty": "hard",
        "tags": ["data_structures", "hash_map", "linked_list"],
    },
    {
        "id": "valid_parentheses_generate",
        "description": (
            "Write a function `generate_parens(n)` that generates all "
            "combinations of n pairs of well-formed parentheses.\n\n"
            "Return them as a sorted list of strings.\n\n"
            "Examples:\n"
            "  generate_parens(1) → ['()']\n"
            "  generate_parens(2) → ['(())', '()()']\n"
            "  generate_parens(3) → ['((()))', '(()())', '(())()', '()(())', '()()()']"
        ),
        "test_code": (
            "assert generate_parens(1) == ['()']\n"
            "assert generate_parens(2) == ['(())', '()()']\n"
            "assert generate_parens(3) == ['((()))', '(()())', '(())()', "
            "'()(())', '()()()']\n"
            "assert len(generate_parens(4)) == 14\n"
            "print('All tests passed!')"
        ),
        "difficulty": "hard",
        "tags": ["backtracking", "recursion"],
    },
]


# ════════════════════════════════════════════════════════════════════
# Lookup Utilities
# ════════════════════════════════════════════════════════════════════

def get_problem(problem_id: str) -> Dict:
    """
    Retrieve a problem by ID.

    Raises KeyError if the problem is not found (fail-fast for
    training pipeline debugging).
    """
    for p in PROBLEMS:
        if p["id"] == problem_id:
            return p
    available = [p["id"] for p in PROBLEMS]
    raise KeyError(f"Problem '{problem_id}' not found. Available: {available}")


def get_problems_by_difficulty(difficulty: str) -> List[Dict]:
    """Get all problems of a given difficulty level."""
    valid = {"easy", "medium", "hard"}
    if difficulty not in valid:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Valid: {valid}")
    return [p for p in PROBLEMS if p["difficulty"] == difficulty]


def get_problems_by_tag(tag: str) -> List[Dict]:
    """Get all problems with a given tag."""
    return [p for p in PROBLEMS if tag in p.get("tags", [])]


def get_all_problem_ids() -> List[str]:
    """List all available problem IDs."""
    return [p["id"] for p in PROBLEMS]


def get_problem_stats() -> Dict:
    """Summary statistics of the problem bank."""
    from collections import Counter
    difficulties = Counter(p["difficulty"] for p in PROBLEMS)
    all_tags = [t for p in PROBLEMS for t in p.get("tags", [])]
    tags = Counter(all_tags)
    return {
        "total": len(PROBLEMS),
        "by_difficulty": dict(difficulties),
        "by_tag": dict(tags),
    }
