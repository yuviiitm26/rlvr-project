from datasets import load_dataset
from typing import List, Dict, Tuple

def get_mbpp_80_20() -> Tuple[List[Dict], List[Dict]]:
    """
    Loads the sanitized MBPP dataset from HuggingFace and strictly limits
    it to 100 problems. Slices into 80 problems for training and 20 for evaluation.
    """
    print("[Data Loader] Loading MBPP 'sanitized' dataset from HuggingFace...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    
    # 1. Take exactly 100 problems
    problems_subset = dataset.select(range(100))
    
    # 2. Split 20/10 for 5.5-hour 1.5B GPU limit (G=8)
    train_dataset = problems_subset.select(range(20))
    eval_dataset = problems_subset.select(range(20, 30))
    
    def format_problem(row) -> Dict:
        # Combine the assertions into an executable test script
        test_code = "\n".join(row["test_list"]) + "\nprint('All tests passed!')"
        
        return {
            "id": f"mbpp_{row['task_id']}",
            "description": row["prompt"],
            "test_code": test_code,
            "test_list": row["test_list"]
        }
        
    train_formatted = [format_problem(row) for row in train_dataset]
    eval_formatted = [format_problem(row) for row in eval_dataset]
    
    print(f"[Data Loader] Loaded {len(train_formatted)} Train problems and {len(eval_formatted)} Eval problems.")
    return train_formatted, eval_formatted
