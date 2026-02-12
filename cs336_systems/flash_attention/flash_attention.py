import torch
import triton
import triton.language as tl

from triton_kernels import flash_attention_forward
from einops import rearrange


class FlashAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,
                q : torch.Tensor, 
                k : torch.Tensor,
                v : torch.Tensor,
                is_causal : bool = False):
        assert q.ndim == 3 
        batch_heads, seq_len, d_model = q.shape, "expected (batch*heads,seq_len,d_model) shape input"
            
        #q,k,v = rearrange([q,k,v], "batch head seq_len d_model -> (batch head) seq_len d_model")
        # will handle head dim before this kernel, data entry will be [(batch*head),seq_len,d_model]
        assert q.is_cuda and k.is_cuda and v.is_cuda, "expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "tensors should be contiguous"
        
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.input_shape = q.shape
        
        N_queries, N_keys, D = q.shape[1], k.shape[1], q.shape[-1]
        scale = 1/D**-0.5
        
        output = torch.empty_like(q)
        log_sum_exp = torch.empty((batch_heads,seq_len), device=q.device, dtype=torch.float32)
        
        flash_attention_forward[(tl.cdiv(seq_len,ctx.Q_TILE_SIZE),batch_heads)](
            q,k,v,
            output,log_sum_exp,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            log_sum_exp.stride(0), log_sum_exp.stride(1),
            N_queries, N_keys, scale, D,
            ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE
        )
        
        ctx.save_for_backward(q, k, v, output, log_sum_exp)
        ctx.scale = scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
        
        
        