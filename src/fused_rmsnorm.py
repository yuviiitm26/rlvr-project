import torch
import triton
import triton.language as tl

@triton.jit
def _fused_rmsnorm_kernel(
    X_ptr, Residual_ptr, Weight_ptr, Out_ptr, Out_Res_ptr,
    stride_x, stride_res, stride_out, stride_ores,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x
    Residual_ptr += row_idx * stride_res
    Out_ptr += row_idx * stride_out
    Out_Res_ptr += row_idx * stride_ores
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(Residual_ptr + offsets, mask=mask, other=0.0)
    
    # Fused Residual Addition
    x_new = x + res
    tl.store(Out_Res_ptr + offsets, x_new, mask=mask)
    
    # RMSNorm
    # Cast to FP32 for variance accumulation to prevent underflow/NaN
    x_new_fp32 = x_new.to(tl.float32)
    variance = tl.sum(x_new_fp32 * x_new_fp32, axis=0) / N
    rsigma = tl.math.rsqrt(variance + eps)
    
    w = tl.load(Weight_ptr + offsets, mask=mask, other=0.0)
    out = (x_new_fp32 * rsigma).to(tl.float16) * w
    tl.store(Out_ptr + offsets, out, mask=mask)

def fused_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    out_res = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 1024:
        raise ValueError("T4 optimized kernel requires BLOCK_SIZE <= 1024")
        
    grid = (M,)
    _fused_rmsnorm_kernel[grid](
        x, residual, weight, out, out_res,
        x.stride(0), residual.stride(0), out.stride(0), out_res.stride(0),
        N, eps,
        num_warps=4,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out, out_res
