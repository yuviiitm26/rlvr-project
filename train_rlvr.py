"""
Multi-Turn 'MURPHY-Style' GRPO Loop (RLVR)

Target: Vertex AI / Kaggle L4/T4 environments.
Upgrades: Triton kernel patching, MBPP 'Easy' Dataset, Multi-Turn Self-Correction.
"""

import os
import torch
from datasets import load_dataset

# 1. APPLY MONKEY PATCH FIRST (Before importing Unsloth/Transformers)
from patch_unsloth import apply_unsloth_patch
apply_unsloth_patch()

from unsloth import FastLanguageModel
from sandbox_grader import PythonJailGrader
from mdp import MultiTurnMDP, RewardConfig
from hf_llm import HuggingFaceLLM
from grpo_trainer import GRPOTrainer
from rewards import compute_grpo_advantages


def load_mbpp_easy(num_problems=10):
    """
    Pulls MBPP from Hugging Face and filters for 'Easy' tier tasks.
    We proxy 'Easy' by selecting tasks with fewer complex assertions 
    that fit comfortably into the 0.5B model's context window.
    """
    print("Loading MBPP dataset from Hugging Face...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    
    problems = []
    for row in dataset:
        # MBPP format: 'text' = description, 'test_list' = array of assertions
        tests = row["test_list"]
        if len(tests) == 0:
            continue
            
        test_code = "\n".join(tests) + "\nprint('All tests passed!')"
        problems.append({
            "id": f"mbpp_{row['task_id']}",
            "description": f"Write a Python function to solve this problem:\n\n{row['text']}",
            "test_code": test_code
        })
        
        if len(problems) >= num_problems:
            break
            
    print(f"Loaded {len(problems)} MBPP Tier 1 problems.")
    return problems


def main():
    print("="*60)
    print("RLVR Phase 3: Multi-Turn GRPO with MBPP and Triton")
    print("="*60)
    
    # 2. Scale Dataset
    problems = load_mbpp_easy(num_problems=10)
    
    # 3. Initialize FastLanguageModel (Patched implicitly)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-bnb-4bit",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Configure LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model, 
        r=16, 
        lora_alpha=16, 
        lora_dropout=0, 
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    grader = PythonJailGrader(use_sandbox=True)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, temperature=0.9)
    
    # 4. Multi-Turn Configuration (MURPHY-Style)
    # Gamma=0.9 strictly prefers 1-shot (1.0) over 2-shot (0.9) over 3-shot (0.81)
    mdp = MultiTurnMDP(
        grader=grader, 
        llm=llm, 
        max_turns=3, 
        reward_config=RewardConfig(discount_gamma=0.9) 
    )
    
    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, group_size=4)
    
    # 5. GRPO Training Loop
    for step, problem in enumerate(problems):
        print(f"\n--- Step {step+1} | MBPP ID: {problem['id']} ---")
        
        trajectories = []
        model.eval() # Freezes weights during Multi-Turn generation
        
        # Rollout G candidates
        for g in range(trainer.group_size):
            # This triggers the test-time compute traceback self-correction loop
            traj = mdp.run_episode(problem)
            trajectories.append(traj)
            
            status = "SOLVED" if traj.solved else "FAILED"
            print(f"  Rollout {g+1}: {traj.num_turns} turns | Reward: {traj.final_reward:.2f} | {status}")
            
        rewards = [t.final_reward for t in trajectories]
        advantages = compute_grpo_advantages(rewards)
        
        # Backward Pass via Patched Autograd
        model.train()
        
        # Get base prompt for logging
        prompt_str = tokenizer.apply_chat_template(
            trajectories[0].turns[0].prompt_messages, tokenize=False, add_generation_prompt=True
        )
        old_log_probs = trainer.compute_old_log_probs(prompt_str, trajectories)
        
        metrics = trainer.train_step(
            trajectories[0].turns[0].prompt_messages, trajectories, advantages, old_log_probs
        )
        print(f"  [GRPO] Loss: {metrics['loss']:.4f} | KL: {metrics['kl']:.4f}")

if __name__ == "__main__":
    main()
