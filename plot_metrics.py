import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="training.log", help="Path to Kaggle training stdout logs")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Log file '{args.log_file}' not found. Please download it from Kaggle outputs.")
        return
        
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    step_rewards = []
    current_step_rewards = []
    
    losses = []
    kls = []
    
    for line in lines:
        # Match: "Rollout 1: 2 turns | Reward: 0.90 | SOLVED"
        reward_match = re.search(r'Reward:\s*([\d\.]+)\s*\|', line)
        if reward_match:
            current_step_rewards.append(float(reward_match.group(1)))
            
        # Match: "  [GRPO] Loss: -0.0123 | KL: 0.0054"
        loss_match = re.search(r'\[GRPO\] Loss:\s*([-\d\.]+)\s*\|\s*KL:\s*([-\d\.]+)', line)
        if loss_match:
            if current_step_rewards:
                step_rewards.append(np.mean(current_step_rewards))
                current_step_rewards = []
                
            losses.append(float(loss_match.group(1)))
            kls.append(float(loss_match.group(2)))
            
    if not losses:
        print("No GRPO metrics found in the log file.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Verifiable Reward
    steps = range(1, len(step_rewards) + 1)
    ax1.plot(steps, step_rewards, marker='o', color='forestgreen', linewidth=2)
    ax1.set_title("Mean Verifiable Reward over Steps")
    ax1.set_xlabel("Training Step (MBPP Problem)")
    ax1.set_ylabel("Mean Discounted Reward (0 to 1)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: GRPO Policy Loss & KL Divergence
    ax2_kl = ax2.twinx()
    
    l1 = ax2.plot(steps, losses, label='Policy Loss', color='crimson', linewidth=2)
    l2 = ax2_kl.plot(steps, kls, label='KL Divergence', color='royalblue', linestyle='--', linewidth=2)
    
    ax2.set_title("GRPO Policy Loss & KL Divergence")
    ax2.set_xlabel("Training Step (MBPP Problem)")
    ax2.set_ylabel("Clipped Surrogate Loss", color='crimson')
    ax2_kl.set_ylabel("KL Approximation (k3)", color='royalblue')
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="upper right")
    
    plt.tight_layout()
    save_path = "telemetry_metrics.png"
    plt.savefig(save_path, dpi=300)
    print(f"Successfully generated telemetry graphs at: {save_path}")

if __name__ == "__main__":
    main()
