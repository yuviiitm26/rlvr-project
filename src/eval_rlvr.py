"""
Evaluation Script for RLVR (Phase 6)
Loads the fine-tuned QLoRA adapters and evaluates on the MBPP eval set.
"""

import os
import torch

from unsloth import FastLanguageModel
from sandbox_grader import PythonJailGrader
from mdp import MultiTurnMDP, RewardConfig
from hf_llm import HuggingFaceLLM
from data_loader import get_mbpp_80_20

def main():
    print("="*60)
    print("RLVR Phase 6: Evaluation")
    print("="*60)
    
    train_problems, eval_problems = get_mbpp_80_20()
    
    # Kaggle mounts the Phase 5 output here
    lora_path = "/kaggle/input/rlvr-project-phase-5/rlvr-project/grpo_saved_lora"
    
    if os.path.exists(lora_path):
        print(f"[Model] Loading fine-tuned RLVR adapters from {lora_path}...")
        model_to_load = lora_path
    else:
        print("[Model] WARNING: LoRA adapters not found. Falling back to base model...")
        model_to_load = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=3072,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Ensure ChatML template
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        
    # Fast inference
    FastLanguageModel.for_inference(model)
        
    # We use greedy decoding for evaluation (temperature=0.0)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, temperature=0.0, max_new_tokens=1024)
    grader = PythonJailGrader(use_sandbox=True)
    
    # We allow up to 3 turns for self-correction
    mdp = MultiTurnMDP(grader=grader, llm=llm, max_turns=3, reward_config=RewardConfig(discount_gamma=0.9, format_reward_weight=0.5))
    
    solved_count = 0
    total = len(eval_problems)
    
    print(f"\n[Eval] Starting Evaluation on {total} problems...")
    
    for i, problem in enumerate(eval_problems):
        print(f"\n--- Eval Problem {i+1}/{total} | MBPP ID: {problem['id']} ---")
        
        problem_prompt = (
            "You are an expert Python programmer. You must strictly follow this format:\n"
            "<think>\n"
            "Step-by-step reasoning goes here...\n"
            "</think>\n"
            "```python\n"
            "# Final working code goes here\n"
            "```\n\n"
            f"Problem: {problem['description']}"
        )
        
        problem_dict = {"id": problem["id"], "description": problem_prompt, "test_code": problem["test_code"]}
        
        # Run 1 trajectory
        traj = mdp.run_episode(problem_dict)
        
        status = "SOLVED" if traj.solved else "FAILED"
        print(f"Result: {traj.num_turns} turns | Reward: {traj.final_reward:.2f} | {status}")
        
        if traj.solved:
            solved_count += 1
            
    pass_rate = (solved_count / total) * 100
    print("="*60)
    print(f"FINAL EVALUATION PASS RATE: {solved_count}/{total} ({pass_rate:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
