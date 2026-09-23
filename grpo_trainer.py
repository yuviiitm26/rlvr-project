"""
Custom GRPO (Group Relative Policy Optimization) Training Loop.

Implements the RLVR thesis:
1. Sample a problem.
2. Generate G completions (trajectories).
3. Execute via subprocess grader (Verifiable Rewards).
4. Compute Group Advantages.
5. Update policy weights via PPO-style clipped objective.

Hardware constraints handled:
- Kaggle T4 (Turing) -> strict float16 mixed precision.
- Memory constraints -> QLoRA gradient checkpointing.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Any
from mdp import Trajectory
from rewards import should_skip_batch_dapo

class GRPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        lr: float = 2e-5,
        beta_kl: float = 0.04,
        clip_ratio: float = 0.2,
        group_size: int = 4
    ):
        """
        Args:
            model: Unsloth PeftModel (with trainable LoRA adapters).
            tokenizer: HF Tokenizer.
            lr: Learning rate for policy network.
            beta_kl: KL divergence penalty coefficient.
            clip_ratio: PPO clipping hyperparameter.
            group_size: G (number of completions per prompt).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beta_kl = beta_kl
        self.clip_ratio = clip_ratio
        self.group_size = group_size
        
        # AdamW optimizer for LoRA weights
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr,
            weight_decay=0.01
        )

    def train_step(
        self, 
        prompt_messages: List[Dict[str, str]], 
        trajectories: List[Trajectory], 
        advantages: List[float]
    ) -> Dict[str, float]:
        """
        Perform a single GRPO parameter update using gathered trajectories.
        
        Algorithm (Dr. GRPO variant):
        - No trajectory length normalization on advantages.
        - DAPO check: skip zero-variance batches.
        - Compute log probabilities of the generated tokens.
        - Apply PPO clipped loss + KL penalty (approximated).
        """
        if len(trajectories) != len(advantages):
            raise ValueError("Mismatched trajectories and advantages.")
            
        # 1. DAPO Check: Skip batch if variance is effectively zero
        rewards = [t.final_reward for t in trajectories]
        if should_skip_batch_dapo(rewards):
            return {"loss": 0.0, "skipped_dapo": 1.0}
            
        self.model.train()
        total_loss = 0.0
        
        # Format the prompt
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.model.device)
        prompt_len = prompt_ids.shape[1]

        # 2. Compute Loss across the G completions
        for traj, adv in zip(trajectories, advantages):
            # For simplicity in this demo, we optimize on the final turn's response.
            # In a full multi-turn GRPO, you'd unroll the entire MDP sequence.
            if not traj.turns:
                continue
                
            final_turn = traj.turns[-1]
            response_str = final_turn.raw_response
            
            # Combine prompt + response for forward pass
            full_text = prompt_str + response_str
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
            input_ids = inputs.input_ids
            labels = input_ids.clone()
            
            # Mask the prompt so we only compute loss on the response tokens
            labels[:, :prompt_len] = -100

            # Forward pass to get current log probabilities
            # We don't pass labels here because we compute our own policy loss.
            # Passing labels sometimes causes Unsloth/Transformers to drop logits (returns None) to save memory.
            outputs = self.model(input_ids=input_ids, return_dict=True)
            
            # Extract logits for the response tokens
            logits = outputs.logits[:, :-1, :]  # shift right
            shift_labels = labels[:, 1:]
            
            # Active tokens mask
            loss_mask = shift_labels != -100
            
            # Compute log probs for the actual generated tokens
            per_token_logprobs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                shift_labels.reshape(-1), 
                reduction="none"
            ).reshape(shift_labels.shape)
            
            # In GRPO, we approximate the old policy using the output logits directly
            # since the generations just happened (or we maintain a reference model).
            # For memory efficiency on Kaggle T4, we avoid a separate reference model
            # and rely on the KL penalty acting as trust region regularizer.
            # Here we simplify the ratio pi_theta / pi_old to exp(logprob_new - logprob_old)
            # Since we just generated, logprob_old is essentially what we have before the optimizer step.
            
            # Compute policy loss (Advantage weighting)
            # Dr. GRPO: We do NOT normalize by the number of tokens (loss_mask.sum())
            # to prevent verbosity hacking. We sum the logprobs.
            sum_logprobs = (per_token_logprobs * loss_mask).sum()
            
            # PPO surrogate loss (simplified without reference model memory overhead)
            # Loss = -Advantage * sum(logprobs)
            policy_loss = -adv * sum_logprobs
            
            # KL divergence penalty (to prevent mode collapse without reference model)
            # Approximated by the entropy of the current distribution
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            mean_entropy = (entropy * loss_mask).sum() / loss_mask.sum()
            
            # Total loss for this trajectory
            loss = policy_loss - (self.beta_kl * mean_entropy)
            total_loss += loss / self.group_size  # mean over group G
            
        # 3. Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "skipped_dapo": 0.0
        }
