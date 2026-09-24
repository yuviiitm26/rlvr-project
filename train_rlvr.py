"""
Multi-Turn 'MURPHY-Style' GRPO Loop (RLVR)
Upgraded with MBPP, System Prompts, and Multi-Turn Loss Masking.
"""

import os
import torch
import torch.nn.functional as F
from torch.amp import autocast

# 1. APPLY MONKEY PATCH FIRST
from patch_unsloth import apply_unsloth_patch
apply_unsloth_patch()

from unsloth import FastLanguageModel
from sandbox_grader import PythonJailGrader
from mdp import MultiTurnMDP, RewardConfig
from hf_llm import HuggingFaceLLM
from grpo_trainer import GRPOTrainer
from rewards import compute_grpo_advantages, should_skip_batch_dapo
from data_loader import get_mbpp_80_20


def build_masked_trajectory(tokenizer, turns):
    """
    Constructs the multi-turn trajectory input_ids and a labels tensor.
    Tokens belonging to the User Prompt and Sandbox Tracebacks are masked with -100.
    Tokens inside the Assistant's <think> and ```python``` blocks receive gradients.
    """
    input_ids = []
    labels = []
    
    for turn in turns:
        # User messages (Initial prompt or Sandbox traceback)
        for msg in turn.prompt_messages:
            if msg["role"] == "user":
                msg_str = tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
                tokens = tokenizer(msg_str, add_special_tokens=False).input_ids
                input_ids.extend(tokens)
                labels.extend([-100] * len(tokens)) # IGNORE_INDEX
                
        # Assistant generation (Gradients ON)
        msg_str = tokenizer.apply_chat_template([{"role": "assistant", "content": turn.raw_response}], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(msg_str, add_special_tokens=False).input_ids
        input_ids.extend(tokens)
        labels.extend(tokens) # LEARN FROM THIS
        
    return torch.tensor([input_ids]), torch.tensor([labels])


def main():
    print("="*60)
    print("RLVR Phase 3: Multi-Turn GRPO with MBPP and -100 Masking")
    print("="*60)
    
    train_problems, eval_problems = get_mbpp_80_20()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-bnb-4bit",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    grader = PythonJailGrader(use_sandbox=True)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, temperature=0.9)
    mdp = MultiTurnMDP(grader=grader, llm=llm, max_turns=3, reward_config=RewardConfig(discount_gamma=0.9))
    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, group_size=4)
    
    # Custom GRPO Loop handling Multi-Turn Masking directly
    for step, problem in enumerate(train_problems):
        print(f"\n--- Train Step {step+1} | MBPP ID: {problem['id']} ---")
        
        # Enforce Prompt Scratchpad
        problem_prompt = (
            "You are an expert Python programmer. \n"
            "You must first analyze the problem step-by-step inside <think> tags. \n"
            "Then, output your final working code inside a ```python ``` block.\n"
            f"Problem: {problem['description']}"
        )
        
        # Pass problem with enforced prompt and preserve id
        problem_dict = {"id": problem["id"], "description": problem_prompt, "test_code": problem["test_code"]}
        
        trajectories = []
        model.eval()
        for g in range(trainer.group_size):
            traj = mdp.run_episode(problem_dict)
            trajectories.append(traj)
            status = "SOLVED" if traj.solved else "FAILED"
            print(f"  Rollout {g+1}: {traj.num_turns} turns | Reward: {traj.final_reward:.2f} | {status}")
            
        rewards = [t.final_reward for t in trajectories]
        if should_skip_batch_dapo(rewards):
            print("  [DAPO] Zero variance, skipping update.")
            continue
            
        advantages = compute_grpo_advantages(rewards)
        
        # --- Multi-Turn MURPHY Backward Pass ---
        model.train()
        trainer.optimizer.zero_grad()
        
        total_loss = 0.0
        total_kl = 0.0
        
        for traj, adv in zip(trajectories, advantages):
            if not traj.turns: continue
            
            input_ids, labels = build_masked_trajectory(tokenizer, traj.turns)
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            
            # Old logprobs (No grad)
            with torch.no_grad():
                old_outputs = model(input_ids=input_ids, return_dict=True)
                old_logits = old_outputs.logits[:, :-1, :]
                old_labels = input_ids[:, 1:]
                old_log_probs = F.log_softmax(old_logits, dim=-1).gather(-1, old_labels.unsqueeze(-1)).squeeze(-1)
            
            # New logprobs (With grad via Patched Autograd)
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, return_dict=True)
                logits = outputs.logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                new_log_probs = F.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                
            # Create Loss Mask (-100 ignored)
            loss_mask = (shift_labels != -100).float()
            
            if loss_mask.sum() == 0: continue
            
            # PPO
            log_ratio = (new_log_probs - old_log_probs) * loss_mask
            ratio = torch.exp(log_ratio)
            
            adv_tensor = torch.tensor(adv, device=model.device, dtype=torch.float32)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - trainer.clip_ratio, 1.0 + trainer.clip_ratio) * adv_tensor
            
            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * loss_mask).sum() / loss_mask.sum()
            
            kl = ((ratio - 1.0 - log_ratio) * loss_mask).sum() / loss_mask.sum()
            
            loss = (policy_loss + trainer.beta_kl * kl) / trainer.group_size
            
            # Scaled Backward
            trainer.scaler.scale(loss).backward()
            
            total_loss += policy_loss.item()
            total_kl += kl.item()
            
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.max_grad_norm)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        
        print(f"  [GRPO] Loss: {total_loss/trainer.group_size:.4f} | KL: {total_kl/trainer.group_size:.4f}")

if __name__ == "__main__":
    main()
