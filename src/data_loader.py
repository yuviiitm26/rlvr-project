from datasets import load_dataset
from typing import List, Dict, Tuple

def get_mbpp_80_20() -> Tuple[List[Dict], List[Dict]]:
    """
    Loads the sanitized MBPP dataset from HuggingFace and strictly limits
    it to 100 problems. Slices into 80 problems for training and 20 for evaluation.
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
