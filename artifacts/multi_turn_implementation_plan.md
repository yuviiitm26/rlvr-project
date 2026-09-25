# Implementation Plan: Multi-Turn "MURPHY-Style" GRPO

## Core Challenge: Gradient Collapse in Multi-Turn RL
In standard GRPO (Group Relative Policy Optimization), the model receives a single prompt and generates a single response. The advantage is applied to the log-probabilities of that response.
In Multi-Turn (Test-Time Compute) RL, the trajectory looks like this:
`Prompt -> Model Attempt 1 -> Traceback -> Model Attempt 2`

If we *only* apply the positive reward to `Attempt 2` (the successful code), the model forgets the context of the traceback and fails to map the relationship between its mistake and the fix. If we apply negative reward to `Attempt 1` and positive to `Attempt 2`, the gradients can destructively interfere, causing gradient collapse and policy degradation (NaNs).

## The MURPHY-Style Solution
To prevent gradient collapse, we treat the **entire conversation history up to the successful code** as a single holistic trajectory. 

### 1. Unified Sequence Log-Probs
Instead of computing loss independently per turn, the `GRPOTrainer` calculates the token-level log-probabilities across the concatenated sequence:
`LogProbs(Response 1) + LogProbs(Response 2)`
The Sandbox Traceback acts as fixed textual context (like the prompt) and is masked out of the loss calculation.

### 2. Discounted Credit Assignment ($\gamma$)
We apply the *final discounted advantage* uniformly across all generated tokens in the sequence. 
*   **1-Shot Success**: Reward = `1.0`
*   **2-Shot Success**: Reward = `0.9` (discounted by $\gamma=0.9$)
*   **Failure**: Reward = `0.0`

**Why this works:**
If the model fails Turn 1, reads the traceback, and fixes it on Turn 2, the combined sequence `(Mistake -> Feedback -> Fix)` earns a positive advantage. This explicitly teaches the network *how to reason and recover from errors*.
However, because $1.0 > 0.9$, the RL engine strictly prefers solving the problem on the first try. This mathematical boundary prevents the model from "reward hacking" (intentionally writing broken code just to receive a traceback and generate more tokens).

### 3. Advantage Normalization
Advantages ($A_i$) are still strictly normalized across the group of $G$ rollouts ($\mu=0, \sigma=1$). 
- If Rollout A gets it on Turn 1 ($R=1.0$), and Rollout B gets it on Turn 2 ($R=0.9$), Rollout A will receive a positive advantage, and Rollout B might receive a slightly negative or zero advantage depending on the group variance. This intensely pushes the policy towards zero-shot proficiency while maintaining the capability for multi-turn debugging.

## Execution Sequence
1. **Initialize**: `patch_unsloth.py` intercepts `Qwen2RMSNorm` via an autograd function invoking Triton.
2. **Load**: `mbpp` easy subset is parsed into assertable strings.
3. **Rollout**: `MultiTurnMDP` executes the generator, runs the sandbox, and appends tracebacks up to 3 times.
4. **Train**: `GRPOTrainer` masks out the prompts/tracebacks and backpropagates the PPO-clipped advantages through the patched Triton kernel.
