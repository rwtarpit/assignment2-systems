import torch

from triton_kernels import flash_attention_forward

class FlashAttentionTRT(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,
                q : torch.Tensor, 
                k : torch.Tensor,
                v : torch.Tensor,
                is_causal : bool = False):
        batch, seq_len, d_model = q.shape
        