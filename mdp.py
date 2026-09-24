"""
Multi-Turn Markov Decision Process (MDP) for RLVR.

Orchestrates the interaction loop between the language model and the
subprocess grader, producing complete trajectories for RL training.

MDP Formalization:
  State   s_t = (problem_description, conversation_history_t)
  Action  a_t = model generates code (text → code extraction)
  Reward  r_t = grader(code, tests) → {0, γ^(t-1)}
  Transition: s_{t+1} = s_t ⊕ (a_t, feedback_t)

Terminal Conditions:
  - Code passes all tests → SUCCESS (reward = γ^(t-1))
  - Turn count exceeds max_turns → FAILURE (reward = 0)

The trajectory T = [(s_1, a_1, r_1), ..., (s_T, a_T, r_T)] is the
fundamental training unit for GRPO. A batch of G trajectories for the
same prompt forms one GRPO training step.

Architecture:
  ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────┐
  │ Problem  │────►│   MDP    │────►│   LLM   │────►│   Grader     │
  │  Bank    │     │  Loop    │◄────│(generate)│     │  (execute)   │
  └─────────┘     └──────────┘     └─────────┘     └──────────────┘
                       │                                    │
                       │         feedback_t                 │
                       ◄────────────────────────────────────┘
                       │
                       ▼
                  ┌──────────┐
                  │Trajectory│ → GRPO training
                  └──────────┘
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from sandbox_grader import ExecutionResult, PythonJailGrader
from rewards import (
    RewardConfig,
    binary_reward,
    composite_reward,
    discounted_reward,
)


# ════════════════════════════════════════════════════════════════════
# Protocols
# ════════════════════════════════════════════════════════════════════

class LLMInterface(Protocol):
    """
    Protocol that any LLM (real or fake) must implement.

    This abstraction lets us swap between:
      - FakeLLM (testing)
      - HuggingFace model (training)
      - API model (evaluation)
    without changing any MDP logic.
    """

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response given a conversation history."""
        ...


# ════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """
    Record of a single turn in the MDP episode.

    Each turn captures the complete state for trajectory replay:
    what the model saw (prompt_messages), what it produced (raw_response),
    what code was extracted, and the execution result.
    """
    turn_number: int
    prompt_messages: List[Dict[str, str]]
    raw_response: str
    extracted_code: str
    execution_result: ExecutionResult
    reward: float
    elapsed_seconds: float


@dataclass
class Trajectory:
    """
    Complete trajectory for one problem episode.

    This is the unit of data consumed by the GRPO training loop.
    A group of G trajectories for the same problem constitutes
    one GRPO optimization step.
    """
    problem_id: str
    problem_description: str
    test_code: str
    turns: List[Turn] = field(default_factory=list)
    solved: bool = False
    final_reward: float = 0.0
    total_elapsed: float = 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON logging / analysis."""
        return {
            "problem_id": self.problem_id,
            "solved": self.solved,
            "num_turns": self.num_turns,
            "final_reward": round(self.final_reward, 4),
            "total_elapsed": round(self.total_elapsed, 3),
            "turns": [
                {
                    "turn": t.turn_number,
                    "passed": t.execution_result.passed,
                    "reward": round(t.reward, 4),
                    "timed_out": t.execution_result.timed_out,
                    "feedback_preview": t.execution_result.formatted_feedback[:200],
                    "code_preview": t.extracted_code[:200],
                }
                for t in self.turns
            ],
        }


# ════════════════════════════════════════════════════════════════════
# Multi-Turn MDP
# ════════════════════════════════════════════════════════════════════

class MultiTurnMDP:
    """
    Runs multi-turn code generation episodes.

    The core loop that connects the language model to the grader.
    Each call to `run_episode()` produces one complete trajectory.

    The MDP loop:
      1. Present the problem to the model (system + user prompt)
      2. Model generates response (reasoning + code)
      3. Extract code from the response
      4. Grade the code via subprocess execution
      5. If passed → terminate with discounted reward
      6. If failed → append formatted feedback to history
      7. If max turns reached → terminate with zero reward
      8. Go to step 2

    Usage:
        grader = SubprocessGrader(use_sandbox=False)
        llm = FakeLLM(strategy="runtime_error_then_fix")
        mdp = MultiTurnMDP(grader, llm, max_turns=4)
        trajectory = mdp.run_episode(problem)
    """

    # System prompt engineering: this shapes the model's output format
    # throughout RL training. Changes here propagate to all episodes.
    SYSTEM_PROMPT = (
        "You are a Python coding assistant. Your task is to write correct "
        "Python code that passes all test assertions.\n\n"
        "RULES:\n"
        "1. Wrap your code in a ```python code block.\n"
        "2. Before writing code, reason about the problem inside "
        "<think></think> tags.\n"
        "3. Write ONLY the solution function(s). Do NOT include test code.\n"
        "4. If you receive error feedback, analyze it carefully and fix "
        "your solution.\n"
    )

    def __init__(
        self,
        grader: PythonJailGrader,
        llm: LLMInterface,
        max_turns: int = 4,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Args:
            grader:        PythonJailGrader instance for code execution.
            llm:           Any object implementing LLMInterface.generate().
            max_turns:     Maximum attempts before episode terminates.
            reward_config: Reward computation configuration.
        """
        self.grader = grader
        self.llm = llm
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def run_episode(self, problem: Dict[str, str]) -> Trajectory:
        """
        Run a complete multi-turn episode for a single problem.

        This is the main entry point. One call = one trajectory.
        In GRPO training, this is called G times per prompt (one per
        completion in the group).

        Args:
            problem: Dict with keys:
                - 'id': Unique problem identifier
                - 'description': Problem statement for the model
                - 'test_code': Assertions to validate the solution
                - 'starter_code': (optional) Code skeleton

        Returns:
            Trajectory containing all turns and final reward.
        """
        trajectory = Trajectory(
            problem_id=problem["id"],
            problem_description=problem["description"],
            test_code=problem["test_code"],
        )

        # Build initial conversation
        messages = self._build_initial_messages(problem)

        episode_start = time.monotonic()

        for turn_num in range(1, self.max_turns + 1):
            turn_start = time.monotonic()

            # ── Step 1: Model generates a response ────────────────
            raw_response = self.llm.generate(messages)

            # ── Step 2: Extract code from the response ────────────
            extracted_code = self._extract_code(raw_response)

            # ── Step 3: Grade the code ────────────────────────────
            exec_result = self.grader.grade(
                extracted_code, problem["test_code"]
            )

            # ── Step 4: Compute reward ────────────────────────────
            reward = self._compute_reward(
                exec_result.passed, turn_num, raw_response
            )

            turn_elapsed = time.monotonic() - turn_start

            # ── Step 5: Record the turn ───────────────────────────
            turn = Turn(
                turn_number=turn_num,
                prompt_messages=list(messages),  # snapshot
                raw_response=raw_response,
                extracted_code=extracted_code,
                execution_result=exec_result,
                reward=reward,
                elapsed_seconds=turn_elapsed,
            )
            trajectory.turns.append(turn)

            # ── Step 6: Check terminal conditions ─────────────────
            if exec_result.passed:
                trajectory.solved = True
                trajectory.final_reward = reward
                break

            # ── Step 7: Append feedback for next turn ─────────────
            # This is the MDP state transition: the model's failed attempt
            # and the grader's feedback become part of the new state.
            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        exec_result, turn_num
                    ),
                }
            )

        trajectory.total_elapsed = time.monotonic() - episode_start

        return trajectory

    # ────────────────────────────────────────────────────────────────
    # Internal: Prompt Construction
    # ────────────────────────────────────────────────────────────────

    def _build_initial_messages(
        self, problem: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Build the initial conversation messages for a problem.

        Structure:
          [system] → establishes output format and constraints
          [user]   → presents the problem
        """
        user_msg = f"## Problem\n\n{problem['description']}"

        if problem.get("starter_code"):
            user_msg += (
                f"\n\n## Starter Code\n"
                f"```python\n{problem['starter_code']}\n```"
            )

        user_msg += (
            "\n\nWrite a Python solution that passes all test assertions. "
            "Think step by step, then provide your code."
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _build_retry_prompt(
        self, exec_result: ExecutionResult, current_turn: int
    ) -> str:
        """
        Build the feedback message appended after a failed attempt.

        This message is critical for multi-turn learning — it's how
        the model receives the "compiler traceback" that drives
        self-correction behavior.
        """
        remaining = self.max_turns - current_turn
        return (
            f"Your solution did not pass. Here is the execution feedback:\n\n"
            f"{exec_result.formatted_feedback}\n\n"
            f"Please fix your solution and try again. "
            f"You have {remaining} attempt(s) remaining."
        )

    # ────────────────────────────────────────────────────────────────
    # Internal: Code Extraction
    # ────────────────────────────────────────────────────────────────

    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from the model's response.

        Extraction priority:
          1. ```python ... ``` blocks (strongest signal)
          2. ``` ... ``` generic code blocks
          3. Raw text with <think> tags stripped (fallback)
        """
        # Priority 1: Explicit Python code blocks
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # Priority 2: Generic code blocks
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Priority 3: Strip think tags, use raw text
        think_stripped = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return think_stripped.strip()

    # ────────────────────────────────────────────────────────────────
    # Internal: Reward Computation
    # ────────────────────────────────────────────────────────────────

    def _compute_reward(
        self, passed: bool, turn: int, raw_response: str
    ) -> float:
        """
        Compute the reward for this turn using the configured strategy.

        Delegates to the rewards module based on RewardConfig settings.
        """
        return composite_reward(
            passed=passed,
            turn=turn,
            response=raw_response,
            config=self.reward_config,
        )
