"""
Custom GRPO (Group Relative Policy Optimization) Training Loop.

Implements the RLVR thesis with proper mixed-precision and PPO clipping:
1. Sample a problem.
2. Generate G completions (trajectories) with model.eval().
3. Execute via PythonJailGrader (Verifiable Rewards).
4. Compute old-policy log-probs for each trajectory.
5. Compute Group Advantages.
6. PPO-clipped surrogate loss with KL penalty.
7. Backprop via GradScaler (FP16 compute, FP32 master weights).

Critical fix: Uses torch.amp.GradScaler to prevent FP16 NaN overflow.
The forward pass runs in FP16 on tensor cores, but gradients accumulate
and optimizer updates in FP32.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from typing import List, Dict, Any, Optional
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
        group_size: int = 4,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: Unsloth PeftModel (with trainable LoRA adapters).
            tokenizer: HF Tokenizer.
            lr: Learning rate for policy network.
            beta_kl: KL divergence penalty coefficient.
            clip_ratio: PPO clipping epsilon.
            group_size: G (number of completions per prompt).
            max_grad_norm: Max gradient norm for clipping.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beta_kl = beta_kl
        self.clip_ratio = clip_ratio
        self.group_size = group_size
        self.max_grad_norm = max_grad_norm

        # AdamW optimizer for LoRA weights
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        # GradScaler for mixed-precision: prevents FP16 underflow/overflow
        # during backward pass by dynamically scaling the loss.
        self.scaler = GradScaler()

    def _compute_log_probs(
        self,
        prompt_str: str,
        response_str: str,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a response given a prompt.
        """
        full_text = prompt_str + response_str
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        input_ids = inputs.input_ids  # [1, seq_len]

        if not hasattr(self, "prompt_token_cache"):
            self.prompt_token_cache = {}
            
        cache_key = hash(prompt_str)
        if cache_key not in self.prompt_token_cache:
            prompt_inputs = self.tokenizer(
                prompt_str, return_tensors="pt", truncation=True, max_length=2048
            )
            self.prompt_token_cache[cache_key] = prompt_inputs.input_ids.shape[1]
            
        prompt_len = self.prompt_token_cache[cache_key]

        # Forward pass
        outputs = self.model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]  # [1, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:]  # [1, seq_len-1]

        # Compute per-token log-probs
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [1, seq_len-1, vocab]
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [1, seq_len-1]

        # Mask: only response tokens (after prompt)
        # prompt_len-1 because of the shift
        response_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        response_start = max(prompt_len - 1, 0)
        response_mask[:, response_start:] = True

        # Return masked log-probs (zeros for prompt tokens)
        masked_log_probs = token_log_probs * response_mask
        return masked_log_probs.squeeze(0), response_mask.squeeze(0)  # [seq_len-1], [seq_len-1]

    @torch.no_grad()
    def compute_old_log_probs(
        self,
        prompt_str: str,
        trajectories: List[Trajectory],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute and freeze old-policy log-probs for all trajectories.

        Called BEFORE the training step, while the model still represents
        the old policy (pi_old). These are used as the denominator in
        the PPO importance ratio.
        """
        self.model.eval()
        old_data = []

        for traj in trajectories:
            if not traj.turns:
                old_data.append(None)
                continue

            final_turn = traj.turns[-1]
            response_str = final_turn.raw_response

            log_probs, mask = self._compute_log_probs(prompt_str, response_str)
            old_data.append({
                "log_probs": log_probs.detach().clone(),
                "mask": mask.detach().clone(),
                "response_str": response_str,
            })

        return old_data

    def train_step(
        self,
        prompt_messages: List[Dict[str, str]],
        trajectories: List[Trajectory],
        advantages: List[float],
        old_log_prob_data: List[Optional[Dict[str, torch.Tensor]]],
    ) -> Dict[str, float]:
        """
        Perform a single GRPO parameter update with PPO clipping.

        Algorithm:
        1. DAPO check: skip zero-variance batches.
        2. For each trajectory:
           a. Compute current log-probs (with gradients)
           b. Compute importance ratio: exp(new_logprob - old_logprob)
           c. Compute PPO clipped surrogate loss
           d. Add KL penalty (Schulman k3 approximation)
        3. Backward pass with GradScaler for mixed-precision safety.
        """
        if len(trajectories) != len(advantages):
            raise ValueError("Mismatched trajectories and advantages.")

        # 1. DAPO Check
        rewards = [t.final_reward for t in trajectories]
        if should_skip_batch_dapo(rewards):
            return {"loss": 0.0, "skipped_dapo": 1.0, "kl": 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        total_loss_val = 0.0
        total_kl = 0.0
        valid_count = 0

        # 2. Compute loss for each trajectory
        for traj, adv, old_data in zip(trajectories, advantages, old_log_prob_data):
            if old_data is None or not traj.turns:
                continue

            response_str = old_data["response_str"]
            old_log_probs = old_data["log_probs"]  # [seq_len-1]
            mask = old_data["mask"]  # [seq_len-1]

            # Compute current policy log-probs (WITH gradients)
            with autocast(device_type="cuda", dtype=torch.float16):
                new_log_probs, new_mask = self._compute_log_probs(
                    prompt_str, response_str
                )

            # Ensure shapes match (truncate to shorter if tokenization differs)
            min_len = min(new_log_probs.shape[0], old_log_probs.shape[0])
            new_lp = new_log_probs[:min_len].float()  # Cast to FP32 for stability
            old_lp = old_log_probs[:min_len].float()
            m = mask[:min_len]

            if m.sum() == 0:
                continue

            # Per-token importance ratio
            log_ratio = new_lp - old_lp
            ratio = torch.exp(log_ratio)

            # PPO clipped surrogate (per-token, then mean over response)
            adv_tensor = torch.tensor(adv, device=ratio.device, dtype=torch.float32)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            ) * adv_tensor
            policy_loss = -torch.min(surr1, surr2)

            # Mean over response tokens (not sum! prevents magnitude explosion)
            policy_loss = (policy_loss * m.float()).sum() / m.float().sum()

            # KL penalty (Schulman's k3 approximation: (ratio - 1) - log_ratio)
            kl_per_token = (ratio - 1.0) - log_ratio
            kl = (kl_per_token * m.float()).sum() / m.float().sum()
            total_kl += kl.item()

            # Total loss for this trajectory
            traj_loss = policy_loss + self.beta_kl * kl
            
            # --- LONGSTRAW: Serialized Backward Pass ---
            # Instead of building a massive computation graph for all trajectories
            # in the group, we backpropagate immediately and clear the tensors.
            scaled_traj_loss = self.scaler.scale(traj_loss / self.group_size)
            scaled_traj_loss.backward()
            
            total_loss_val += (traj_loss.item() / self.group_size)
            valid_count += 1
            
            # Explicitly delete activation tensors to free VRAM for the next iteration
            del new_log_probs, new_lp, old_lp, log_ratio, ratio
            del surr1, surr2, policy_loss, kl_per_token, kl, traj_loss, scaled_traj_loss

        if valid_count == 0:
            return {"loss": 0.0, "skipped_dapo": 1.0, "kl": 0.0}

        # 3. Optimizer Step (after all trajectories have accumulated gradients)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Adaptive KL Controller
        avg_kl = total_kl / max(valid_count, 1)
        target_kl = 0.02
        if avg_kl > target_kl * 1.5:
            self.beta_kl *= 1.2
        elif avg_kl < target_kl * 0.5:
            self.beta_kl *= 0.8
        self.beta_kl = max(0.001, min(self.beta_kl, 0.1))

        return {
            "loss": total_loss_val,
            "skipped_dapo": 0.0,
            "kl": total_kl / max(valid_count, 1),
        }
