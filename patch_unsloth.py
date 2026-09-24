import torch
import triton
import transformers
from fused_rmsnorm import fused_rmsnorm

class FusedRMSNormAutograd(torch.autograd.Function):
    """
    Autograd wrapper for our Phase 3 Triton Fused RMSNorm kernel.
    Allows PyTorch's backward pass to flow through the custom kernel.
    """
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        # 1. Forward pass strictly via Triton
        out, out_res = fused_rmsnorm(x, residual, weight, eps)
        
        # Save tensors for the analytical backward pass
        ctx.save_for_backward(x, residual, weight, out_res)
        ctx.eps = eps
        return out, out_res

    @staticmethod
    def backward(ctx, grad_out, grad_out_res):
        x, residual, weight, out_res = ctx.saved_tensors
        eps = ctx.eps
        
        # 2. PyTorch Fallback Backward
        # In a fully optimized system, we would write a @triton.jit backward kernel.
        # For this Phase, we enable standard PyTorch autograd on the forward variables 
        # to guarantee mathematical correctness during the QLoRA backward pass.
        with torch.enable_grad():
            x_var = x.detach().requires_grad_(True)
            res_var = residual.detach().requires_grad_(True)
            w_var = weight.detach().requires_grad_(True)
            
            # Recompute graph
            x_new = x_var + res_var
            variance = x_new.to(torch.float32).pow(2).mean(-1, keepdim=True)
            out = x_new * torch.rsqrt(variance + eps) * w_var
            out = out.to(x.dtype)
            
            # Compute gradients analytically
            out.backward(grad_out)
            
        return x_var.grad, res_var.grad, w_var.grad, None


def apply_unsloth_patch():
    """
    Monkey-patches Qwen2's modeling to intercept RMSNorm operations
    and route them through our Triton kernel. MUST be called before
    FastLanguageModel.from_pretrained().
    """
    import transformers.models.qwen2.modeling_qwen2 as hf_qwen2
    try:
        import unsloth.models.qwen2 as unsloth_qwen2
        has_unsloth = True
    except ImportError:
        has_unsloth = False

    class PatchedQwen2RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states, residual=None):
            # If no residual is provided, pass a zero-tensor placeholder
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            
            # Call Triton Kernel via Autograd
            out, out_res = FusedRMSNormAutograd.apply(
                hidden_states, residual, self.weight, self.variance_epsilon
            )
            
            # Unsloth/HF expects a tuple if residual was provided, else tensor
            return out

    # Override the classes in memory
    hf_qwen2.Qwen2RMSNorm = PatchedQwen2RMSNorm
    if has_unsloth:
        unsloth_qwen2.Qwen2RMSNorm = PatchedQwen2RMSNorm
        
    print("[Systems Engine] 🐒 Successfully monkey-patched Qwen2RMSNorm with Triton Fused Kernel.")
