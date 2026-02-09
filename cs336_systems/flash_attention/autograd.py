import torch
from torch.autograd import Function

class FlashAttention(Function):
    "autograd implementation of flash attention"
    
    @staticmethod
    def forward(ctx,
                q : torch.Tensor, # (seq_len,d_model)
                k : torch.Tensor,
                v : torch.Tensor,
                tile_size : int = 16,
                is_causal : bool = True):
        
        tile_q = torch.arange(0,q.shape[0],tile_size)
        d_model = q.shape[-1]
        
        for i in tile_q:
            current_q = q[i:i+1,...]
            output = torch.zeros_like(current_q)
            l1 = torch.zeros(tile_size)
            max_el = -torch.inf(tile_size)
            
            for j in tile_q:
                current_k = k[j:j+1,...]
                current_v = v[j:j+1,...]
                
                mat_mul = (current_q@current_k.T)/d_model
                row_max = torch.max(mat_mul,dim=-1)
                max_el = torch.max(max_el,row_max)
                p1 = torch.exp(mat_mul - max_el)  
                l1 =   
        