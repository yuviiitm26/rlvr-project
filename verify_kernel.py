import torch
import traceback
import sys

def run_verification():
    print("="*60)
    print("Phase 3: Triton Kernel Verification (Fused RMSNorm)")
    print("="*60)
    
    try:
        from fused_rmsnorm import fused_rmsnorm
    except ImportError as e:
        print(f"FAILED TO IMPORT: {e}")
        return
        
    # Check if Triton is actually available (Linux/CUDA required)
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on this host. Triton requires an NVIDIA GPU.")
        print("Please execute this verification script directly on the Google Colab instance.")
        return
        
    M, N = 128, 1024
    print(f"Allocating Test Tensors: [M={M}, N={N}] in FP16...")
    
    torch.manual_seed(42)
    x = torch.randn((M, N), dtype=torch.float16, device='cuda')
    residual = torch.randn((M, N), dtype=torch.float16, device='cuda')
    weight = torch.ones((N,), dtype=torch.float16, device='cuda')
    eps = 1e-6

    # --- Baseline PyTorch ---
    print("Computing PyTorch Baseline...")
    x_new_torch = x + residual
    variance = x_new_torch.to(torch.float32).pow(2).mean(-1, keepdim=True)
    out_torch = x_new_torch * torch.rsqrt(variance + eps) * weight
    out_torch = out_torch.to(torch.float16)

    # --- Triton Kernel ---
    print("Computing Triton Fused Kernel (num_warps=4, block_size=1024)...")
    try:
        out_triton, x_new_triton = fused_rmsnorm(x, residual, weight, eps)
        
        print("Verifying Residual Addition...")
        assert torch.allclose(x_new_triton, x_new_torch, atol=1e-3, rtol=1e-3), "Residual mismatch!"
        
        print("Verifying RMSNorm Output...")
        assert torch.allclose(out_triton, out_torch, atol=1e-3, rtol=1e-3), "RMSNorm mismatch!"
        
        print("\n[SUCCESS] Triton kernel mathematically equivalent to PyTorch baseline!")
    except Exception as e:
        print(f"\n[FAILED] Triton execution crashed:")
        traceback.print_exc()

if __name__ == '__main__':
    run_verification()
