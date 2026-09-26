"""
RLVR Evaluation Script.
Loads the trained QLoRA adapters from the Phase 2 training run and evaluates
them on the 10 MBPP hold-out eval problems.
"""

import os
import sys
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
import problems
from sandbox_grader import PythonJailGrader

def main():
    print("="*60)
    print("🚀 Starting RLVR Evaluation")
    print("="*60)

    # 1. Load Eval Problems from MBPP
    print("[Data] Loading MBPP 'sanitized' validation split from HuggingFace...")
    dataset = load_dataset("mbpp", "sanitized")
    eval_dataset = dataset["validation"].select(range(10)) # Take first 10 for quick eval
    
    eval_problems = []
    for row in eval_dataset:
        eval_problems.append({
            "id": f"mbpp_{row['task_id']}",
            "description": row["text"],
            "test_code": "\n".join(row["test_list"])
        })
    print(f"[Data] Loaded {len(eval_problems)} MBPP Evaluation Problems.")

    # 2. Load Model & Trained LoRA Adapters
    print("[Model] Loading Unsloth base model + trained RLVR adapters...")
    lora_path = "./grpo_saved_lora"
    
    if not os.path.exists(lora_path):
        print(f"ERROR: Could not find trained adapters at {lora_path}!")
        print("Please ensure the training notebook is attached as a Data Source.")
        sys.exit(1)

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path, # Load directly from the saved LoRA directory!
        max_seq_length = max_seq_length,
        dtype = torch.float16,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    grader = PythonJailGrader()

    total_passed = 0

    print("="*60)
    print("🧠 Running Evaluation")
    print("="*60)

    for i, problem in enumerate(eval_problems):
        print(f"\n--- Eval Problem {i+1}/{len(eval_problems)}: {problem['id']} ---")
        
        prompt = (
            "You are an expert Python programmer. You must strictly follow this format:\n"
            "<think>\n"
            "Step-by-step reasoning goes here...\n"
            "</think>\n"
            "```python\n"
            "# Final working code goes here\n"
            "```\n\n"
            f"Problem: {problem['description']}"
        )

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract code block
        code_start = response.find("```python")
        code_end = response.rfind("```")
        if code_start != -1 and code_end != -1 and code_end > code_start:
            code = response[code_start + 9:code_end].strip()
        else:
            code = response.strip()

        # Grade it
        result = grader.grade(code, problem["test_code"])
        
        if result.passed:
            print("✅ PASSED!")
            total_passed += 1
        else:
            print(f"❌ FAILED. Error: {result.error_type}")
            print(f"Feedback: {result.formatted_feedback.splitlines()[0]}")

    print("="*60)
    print(f"🏆 Final Eval Accuracy: {total_passed}/{len(eval_problems)} ({(total_passed/len(eval_problems))*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
