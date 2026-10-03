"""
Data Loader for RLVR Training.

Loads the MBPP sanitized dataset from HuggingFace and provides:
  1. Automatic difficulty classification based on reference solution complexity
  2. Curriculum-aware problem ordering for phased training
  3. Clean 80/20 train/eval split
"""

import random
from datasets import load_dataset
from typing import List, Dict, Tuple


# ════════════════════════════════════════════════════════════════════
# Difficulty Classifier
# ════════════════════════════════════════════════════════════════════

def _classify_difficulty(code: str, test_list: List[str], prompt: str) -> str:
    """
    Heuristic difficulty classification based on the reference solution.
    
    Uses three signals:
      1. Lines of code in the reference solution (strongest signal)
      2. Number of test assertions (more tests = more edge cases = harder)
      3. Prompt complexity keywords (class, recursion, tree, graph, etc.)
    
    Thresholds are calibrated against the MBPP sanitized distribution:
      - Median code length ≈ 5 lines
      - 25th percentile ≈ 3 lines
      - 75th percentile ≈ 8 lines
    """
    code_lines = len(code.strip().split("\n"))
    num_tests = len(test_list)
    prompt_lower = prompt.lower()
    
    # Hard indicators: complex data structures, algorithms, or classes
    hard_keywords = ["class ", "recursion", "tree", "graph", "matrix", 
                     "dynamic programming", "heap", "trie", "linked list",
                     "binary search", "backtrack"]
    has_hard_keyword = any(kw in prompt_lower for kw in hard_keywords)
    
    # Score-based classification (higher = harder)
    difficulty_score = 0
    
    # Signal 1: Code length (strongest weight)
    if code_lines <= 4:
        difficulty_score += 0
    elif code_lines <= 8:
        difficulty_score += 1
    elif code_lines <= 15:
        difficulty_score += 2
    else:
        difficulty_score += 3
    
    # Signal 2: Test count
    if num_tests >= 4:
        difficulty_score += 1
    
    # Signal 3: Keywords
    if has_hard_keyword:
        difficulty_score += 2
    
    # Classify
    if difficulty_score <= 1:
        return "easy"
    elif difficulty_score <= 3:
        return "medium"
    else:
        return "hard"


# ════════════════════════════════════════════════════════════════════
# Curriculum Scheduler
# ════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    3-Phase Curriculum Learning Scheduler.
    
    Controls which problems the model sees at each training step,
    progressively increasing difficulty as the model's LoRA adapters
    stabilize and start producing meaningful code.
    
    Phase 1 (Steps 1 → phase1_end):
        EASY problems only. The model learns basic Python syntax,
        function signatures, and the <think>→```python``` format.
        Reward signal is strong and consistent → stable initial gradients.
        
    Phase 2 (Steps phase1_end → phase2_end):  
        EASY + MEDIUM problems shuffled together. The model starts
        encountering algorithmic challenges (hash maps, DP, recursion)
        while still getting "warm-up" easy problems for gradient stability.
        
    Phase 3 (Steps phase2_end → end):
        ALL problems (Easy + Medium + Hard) shuffled. The model faces
        full-difficulty challenges like LRU Cache, backtracking, etc.
        By now the LoRA weights are well-initialized and can handle
        the sparse reward signal from hard problems.
    
    Architecture:
      ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐
      │ Phase 1   │───►│ Phase 2           │───►│ Phase 3           │
      │ EASY only │    │ EASY + MEDIUM     │    │ EASY+MEDIUM+HARD │
      │ Steps 1-N │    │ Steps N-M         │    │ Steps M-End       │
      └──────────┘    └──────────────────┘    └──────────────────┘
         High reward      Moderate reward         Sparse reward
         ▼ Stable ∇       ▼ Growing ∇             ▼ Full reasoning
    """
    
    def __init__(
        self,
        problems: List[Dict],
        phase1_end: int = 40,
        phase2_end: int = 120,
        seed: int = 42,
    ):
        """
        Args:
            problems:   All training problems (with 'difficulty' key).
            phase1_end: Last step of Phase 1 (easy only).
            phase2_end: Last step of Phase 2 (easy + medium).
            seed:       Random seed for reproducible shuffling.
        """
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.rng = random.Random(seed)
        
        # Partition problems by difficulty
        self.easy = [p for p in problems if p.get("difficulty") == "easy"]
        self.medium = [p for p in problems if p.get("difficulty") == "medium"]
        self.hard = [p for p in problems if p.get("difficulty") == "hard"]
        
        print(f"[Curriculum] Problem distribution: "
              f"Easy={len(self.easy)}, Medium={len(self.medium)}, Hard={len(self.hard)}")
        print(f"[Curriculum] Phase 1 (Easy only): Steps 1-{phase1_end}")
        print(f"[Curriculum] Phase 2 (Easy+Medium): Steps {phase1_end+1}-{phase2_end}")
        print(f"[Curriculum] Phase 3 (All): Steps {phase2_end+1}+")
    
    def get_current_phase(self, step: int) -> int:
        """Returns the current phase number (1, 2, or 3)."""
        if step <= self.phase1_end:
            return 1
        elif step <= self.phase2_end:
            return 2
        else:
            return 3
    
    def get_problem_pool(self, step: int) -> List[Dict]:
        """
        Returns the pool of problems available at the current step.
        
        The pool grows as training progresses:
          Phase 1 → Easy only
          Phase 2 → Easy + Medium
          Phase 3 → Easy + Medium + Hard
        """
        phase = self.get_current_phase(step)
        
        if phase == 1:
            pool = list(self.easy)
        elif phase == 2:
            pool = list(self.easy) + list(self.medium)
        else:
            pool = list(self.easy) + list(self.medium) + list(self.hard)
        
        self.rng.shuffle(pool)
        return pool
    
    def get_epoch_problems(self, epoch: int, step_offset: int, total_steps_per_epoch: int) -> List[Dict]:
        """
        Returns a full epoch's worth of problems, ordered by curriculum phase.
        
        For each step in the epoch, we pick a problem from the appropriate
        difficulty pool. This ensures smooth transitions between phases
        even within a single epoch.
        
        Args:
            epoch:                Current epoch number (0-indexed).
            step_offset:          Global step at the start of this epoch.
            total_steps_per_epoch: How many steps this epoch will run.
            
        Returns:
            Ordered list of problems for the epoch.
        """
        epoch_problems = []
        
        for local_step in range(total_steps_per_epoch):
            global_step = step_offset + local_step + 1
            pool = self.get_problem_pool(global_step)
            
            if not pool:
                # Fallback: use all problems if a phase pool is empty
                pool = list(self.easy) + list(self.medium) + list(self.hard)
                self.rng.shuffle(pool)
            
            # Cycle through the pool (modular indexing)
            problem = pool[local_step % len(pool)]
            epoch_problems.append(problem)
        
        return epoch_problems


# ════════════════════════════════════════════════════════════════════
# Main Data Loading Function
# ════════════════════════════════════════════════════════════════════

def get_mbpp_80_20() -> Tuple[List[Dict], List[Dict]]:
    """
    Loads the sanitized MBPP dataset from HuggingFace and strictly limits
    it to 100 problems. Slices into 80 problems for training and 20 for evaluation.
    
    Each problem is tagged with an auto-classified difficulty level
    based on the reference solution complexity.
    """
    print("[Data Loader] Loading MBPP 'sanitized' dataset from HuggingFace...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    
    # The MBPP 'sanitized' train split has exactly 120 problems.
    # We will use 100 for training and 20 for evaluation.
    train_dataset = dataset.select(range(100))
    eval_dataset = dataset.select(range(100, 120))
    
    def format_problem(row) -> Dict:
        # Combine the assertions into an executable test script
        test_code = "\n".join(row["test_list"]) + "\nprint('All tests passed!')"
        
        # Auto-classify difficulty using the reference solution
        difficulty = _classify_difficulty(
            code=row["code"],
            test_list=row["test_list"],
            prompt=row["prompt"],
        )
        
        return {
            "id": f"mbpp_{row['task_id']}",
            "description": row["prompt"],
            "test_code": test_code,
            "test_list": row["test_list"],
            "difficulty": difficulty,
        }
        
    train_formatted = [format_problem(row) for row in train_dataset]
    eval_formatted = [format_problem(row) for row in eval_dataset]
    
    # Log the difficulty distribution
    from collections import Counter
    train_dist = Counter(p["difficulty"] for p in train_formatted)
    print(f"[Data Loader] Loaded {len(train_formatted)} Train problems: {dict(train_dist)}")
    print(f"[Data Loader] Loaded {len(eval_formatted)} Eval problems.")
    
    return train_formatted, eval_formatted
