"""
Training Entrypoint for RLVR.

Combines the Unsloth QLoRA model, HuggingFaceLLM, Multi-Turn MDP, 
and GRPO Trainer into the main training loop.

Phase 2 Stabilization:
  - Mixed-precision training (FP16 compute, FP32 master weights)
  - Async batch sandbox evaluation (ThreadPoolExecutor)
  - Proper generation/training phase separation
  - Curriculum learning (Tier 1 easy → Tier 2 medium)

Designed to execute on Kaggle Dual T4s or Google Colab L4s.
"""

import os
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unsloth import FastLanguageModel

from sandbox_grader import PythonJailGrader
from mdp import MultiTurnMDP
from hf_llm import HuggingFaceLLM
from problems import PROBLEMS, get_problems_by_difficulty
from grpo_trainer import GRPOTrainer
from rewards import compute_grpo_advantages, RewardConfig

def setup_unsloth_model(
    model_name="unsloth/Qwen2.5-0.5B-bnb-4bit",
    max_seq_length=2048,
):
    """
    Load the pre-quantized Unsloth model and apply QLoRA adapters.

    Why unsloth/Qwen2.5-0.5B-bnb-4bit instead of Qwen/Qwen2.5-0.5B:
      - The 'unsloth/' prefix pulls a natively 4-bit bitsandbytes checkpoint.
      - This skips on-the-fly quantization, saving ~2GB peak VRAM on T4.
      - The model weights are already packed in NF4 format, so load is ~2x faster.

    Why float16 (not bfloat16):
      - T4 GPUs (Turing SM75) do NOT have native bfloat16 tensor cores.
      - Using bf16 on T4 silently falls back to fp32 emulation, doubling memory.
      - Forcing fp16 ensures real tensor core acceleration and correct gradients.

    Why max_seq_length=2048:
      - Our MDP produces multi-turn conversations with Python tracebacks.
      - 2048 tokens gives enough room for: system prompt (~200) + 3 turns of
        code generation (~300 each) + traceback feedback (~200 each) = ~1700 tokens.
      - Going higher wastes KV cache memory during GRPO's G parallel generations.
    """
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


def generate_rollouts(mdp, problem, group_size):
    """
    Generate G rollouts for a single problem.
    
    Uses ThreadPoolExecutor for async sandbox evaluation.
    The MDP's internal subprocess calls are I/O-bound,
    so threading gives near-linear speedup.
    """
    trajectories = []
    
    # We use threading because the bottleneck is subprocess I/O (sandbox execution),
    # not CPU compute. Python's GIL doesn't block subprocess.run().
    with ThreadPoolExecutor(max_workers=group_size) as executor:
        futures = [
            executor.submit(mdp.run_episode, problem)
            for _ in range(group_size)
        ]
        for future in as_completed(futures):
            traj = future.result()
            trajectories.append(traj)
    
    return trajectories


def main():
    print("=" * 60)
    print("RLVR Phase 2: Stabilized GRPO Training")
    print("=" * 60)
    
    # 1. Initialize hardware constraints & models
    # Using pre-quantized Unsloth checkpoint (natively 4-bit, no on-the-fly quant)
    model, tokenizer = setup_unsloth_model(
        model_name="unsloth/Qwen2.5-0.5B-bnb-4bit"
    )
    
    # 2. Setup the MDP Components
    use_sandbox = os.path.exists("/tmp")
    grader = PythonJailGrader(use_sandbox=use_sandbox)
    
    llm = HuggingFaceLLM(
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.9  # High temperature for exploration diversity
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
        group_size=4,  # G=4 completions per prompt
        lr=2e-5,
        beta_kl=0.04,
        clip_ratio=0.2,
    )
    
    # 3. Curriculum: Start with Tier 1 (easy) problems only
    # A 0.5B base model needs achievable targets to bootstrap learning.
    # Medium/hard problems are added after sustained reward lift.
    tier1_problems = get_problems_by_difficulty("easy")
    
    print(f"\nCurriculum: {len(tier1_problems)} Tier 1 (easy) problems")
    print(f"Group Size: G={trainer.group_size}")
    print(f"Max Turns: {mdp.max_turns}")
    print(f"Sandbox: {'ENABLED' if use_sandbox else 'DISABLED'}")
    
    # 4. The GRPO Training Loop
    epochs = 1
    total_steps = 0
    total_rewards = []
    
    print("\n" + "=" * 60)
    print("Starting GRPO Training Loop...")
    print("=" * 60)
    
    for epoch in range(epochs):
        for problem in tier1_problems:
            step_start = time.time()
            print(f"\n--- Step {total_steps+1}: Problem '{problem['id']}' ---")
            
            # ═══════════════════════════════════════════════════════
            # Phase A: GENERATION (model.eval(), no gradients)
            # ═══════════════════════════════════════════════════════
            model.eval()
            
            trajectories = generate_rollouts(mdp, problem, trainer.group_size)
            rewards = [t.final_reward for t in trajectories]
            
            for i, traj in enumerate(trajectories):
                status = "✓ SOLVED" if traj.solved else "✗ FAILED"
                print(f"  Rollout {i+1}/{trainer.group_size}: "
                      f"Turns={traj.num_turns} | Reward={traj.final_reward:.2f} | {status}")
            
            # Compute advantages
            advantages = compute_grpo_advantages(rewards)
            print(f"  Rewards:    {[round(r, 2) for r in rewards]}")
            print(f"  Advantages: {[round(a, 2) for a in advantages]}")
            
            # ═══════════════════════════════════════════════════════
            # Phase B: TRAINING (model.train(), with gradients)
            # ═══════════════════════════════════════════════════════
            
            # Compute old-policy log-probs BEFORE weight update
            initial_messages = trajectories[0].turns[0].prompt_messages
            prompt_str = tokenizer.apply_chat_template(
                initial_messages, tokenize=False, add_generation_prompt=True
            )
            old_log_prob_data = trainer.compute_old_log_probs(
                prompt_str, trajectories
            )
            
            # Perform GRPO update step
            metrics = trainer.train_step(
                initial_messages, trajectories, advantages, old_log_prob_data
            )
            
            step_time = time.time() - step_start
            
            if metrics["skipped_dapo"] > 0:
                print(f"  [DAPO] Zero variance batch — skipping gradient update.")
            else:
                print(f"  [GRPO] Loss: {metrics['loss']:.4f} | "
                      f"KL: {metrics['kl']:.4f} | "
                      f"Time: {step_time:.1f}s")
            
            total_steps += 1
            total_rewards.extend(rewards)
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    if total_rewards:
        import numpy as np
        print(f"  Total Steps: {total_steps}")
        print(f"  Mean Reward: {np.mean(total_rewards):.3f}")
        print(f"  Pass Rate:   {sum(1 for r in total_rewards if r > 0) / len(total_rewards):.1%}")
    
    print("  Model weights updated via Verifiable Rewards.")

if __name__ == "__main__":
    main()
