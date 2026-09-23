"""
Training Entrypoint for RLVR.

Combines the Unsloth QLoRA model, HuggingFaceLLM, Multi-Turn MDP, 
and GRPO Trainer into the main training loop.

Designed to execute on Kaggle Dual T4s or Google Colab L4s.
"""

import os
import torch
from unsloth import FastLanguageModel

from sandbox_grader import PythonJailGrader
from mdp import MultiTurnMDP
from hf_llm import HuggingFaceLLM
from problems import PROBLEMS
from grpo_trainer import GRPOTrainer
from rewards import compute_grpo_advantages, RewardConfig

def setup_unsloth_model(model_name="Qwen/Qwen2.5-1.5B", max_seq_length=2048):
    """Load the base model with 4-bit quantization and LoRA adapters."""
    print(f"Loading {model_name} via Unsloth...")
    
    # Turing GPUs (T4) do NOT support bfloat16 natively. 
    # Must use float16 to prevent underflow/NaNs in gradients.
    dtype = torch.float16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    
    # Setup LoRA adapters for QLoRA training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )
    
    # Ensure correct pad token for Qwen
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Base models lack a chat_template. Inject a standard ChatML template.
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        
    return model, tokenizer

def main():
    print("=" * 60)
    print("RLVR Phase 2: Distributed GRPO Training")
    print("=" * 60)
    
    # 1. Initialize hardware constraints & models
    model, tokenizer = setup_unsloth_model(model_name="Qwen/Qwen2.5-0.5B")
    
    # 2. Setup the MDP Components
    # Assuming Kaggle environment; use_sandbox=True enforces the jail.
    # On Windows local dev, change to False.
    use_sandbox = os.path.exists("/tmp") 
    grader = PythonJailGrader(use_sandbox=use_sandbox)
    
    llm = HuggingFaceLLM(
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    mdp = MultiTurnMDP(
        grader=grader,
        llm=llm,
        max_turns=3,
        reward_config=RewardConfig(discount_gamma=0.9)
    )
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        group_size=4 # G=4 completions per prompt
    )
    
    # 3. The GRPO Training Loop
    epochs = 1
    
    # For demo purposes, we train on the first two easy problems
    training_problems = [p for p in PROBLEMS if p["difficulty"] == "easy"][:2]
    
    print("\nStarting GRPO Loop...")
    for epoch in range(epochs):
        for problem in training_problems:
            print(f"\n--- Training on Problem: {problem['id']} ---")
            
            # G completions per prompt
            trajectories = []
            rewards = []
            
            # We freeze the model during generation (handled by hf_llm.py eval mode)
            for g in range(trainer.group_size):
                print(f"  Generating completion {g+1}/{trainer.group_size}...")
                traj = mdp.run_episode(problem)
                trajectories.append(traj)
                rewards.append(traj.final_reward)
                
                status = "✓ SOLVED" if traj.solved else "✗ FAILED"
                print(f"    Turns: {traj.num_turns} | Reward: {traj.final_reward:.2f} | {status}")
                
            # Compute advantages
            advantages = compute_grpo_advantages(rewards)
            print(f"  Group Rewards:    {[round(r, 2) for r in rewards]}")
            print(f"  Group Advantages: {[round(a, 2) for a in advantages]}")
            
            # Perform GRPO update step
            # Note: The MDP automatically builds the system prompt in traj.turns[0].prompt_messages
            initial_messages = trajectories[0].turns[0].prompt_messages
            
            metrics = trainer.train_step(initial_messages, trajectories, advantages)
            
            if metrics["skipped_dapo"] > 0:
                print("  [DAPO] Zero variance batch detected. Skipping gradient update.")
            else:
                print(f"  [GRPO] Gradient step applied. Loss: {metrics['loss']:.4f}")

    print("\nTraining run complete!")
    print("Model weights updated via Verifiable Rewards.")

if __name__ == "__main__":
    main()
