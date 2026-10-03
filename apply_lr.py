with open('src/grpo_trainer.py', 'r', encoding='utf-8') as f:
    grpo_content = f.read()

# 1. Add imports to grpo_trainer.py
if "from torch.optim.lr_scheduler" not in grpo_content:
    grpo_content = grpo_content.replace(
        "from torch.optim import AdamW",
        "from torch.optim import AdamW\nfrom torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR"
    )

# 2. Modify __init__ signature and body
old_init_sig = """    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        lr: float = 2e-5,
        beta_kl: float = 0.04,
        clip_ratio: float = 0.2,
        group_size: int = 4,
        max_grad_norm: float = 1.0,
    ):"""

new_init_sig = """    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        lr: float = 2e-5,
        beta_kl: float = 0.04,
        clip_ratio: float = 0.2,
        group_size: int = 4,
        max_grad_norm: float = 1.0,
        total_steps: int = 200,
        warmup_steps: int = 20,
    ):"""
grpo_content = grpo_content.replace(old_init_sig, new_init_sig)

old_init_body = """        # GradScaler for mixed-precision: prevents FP16 underflow/overflow
        # during backward pass by dynamically scaling the loss.
        self.scaler = GradScaler()"""

new_init_body = """        # GradScaler for mixed-precision: prevents FP16 underflow/overflow
        # during backward pass by dynamically scaling the loss.
        self.scaler = GradScaler()
        
        # Learning Rate Schedulers (Warmup + Cosine Decay)
        # Prevents early chaotic gradients and late overshooting.
        warmup = LinearLR(self.optimizer, start_factor=0.05, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 0.1)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])"""
grpo_content = grpo_content.replace(old_init_body, new_init_body)

# 3. Step the scheduler in train_step
old_step = """        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Adaptive KL Controller"""

new_step = """        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        # Adaptive KL Controller"""
grpo_content = grpo_content.replace(old_step, new_step)

with open('src/grpo_trainer.py', 'w', encoding='utf-8') as f:
    f.write(grpo_content)

print("Updated grpo_trainer.py successfully.")

# ==========================================

with open('src/train_rlvr.py', 'r', encoding='utf-8') as f:
    train_content = f.read()

# Update train_rlvr.py to pass total_steps and warmup_steps
old_trainer_init = 'trainer = GRPOTrainer(model=model, tokenizer=tokenizer, group_size=4, lr=5e-5) # Reduced group_size from 8 to 4 to prevent OOM'
new_trainer_init = '''    # Calculate steps for LR Scheduler
    EPOCHS = 2
    total_steps_per_epoch = len(train_problems)
    total_training_steps = EPOCHS * total_steps_per_epoch
    warmup_steps = max(10, int(total_training_steps * 0.1)) # 10% warmup
    
    trainer = GRPOTrainer(
        model=model, 
        tokenizer=tokenizer, 
        group_size=4, 
        lr=5e-5,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps
    )'''
train_content = train_content.replace(old_trainer_init, new_trainer_init)

# Remove the old EPOCHS = 2 line since we moved it up
train_content = train_content.replace('    EPOCHS = 2\n    global_step = 0\n    total_steps_per_epoch = len(train_problems)', '    global_step = 0')

with open('src/train_rlvr.py', 'w', encoding='utf-8') as f:
    f.write(train_content)
    
print("Updated train_rlvr.py successfully.")
