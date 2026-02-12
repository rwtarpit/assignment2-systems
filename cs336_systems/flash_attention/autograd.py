import torch
from torch.autograd import Function
from einops import rearrange, einsum, reduce

class FlashAttention_Torch(Function):
    "autograd implementation of flash attention"
    
    @staticmethod
    def forward(ctx,
                q : torch.Tensor, # (b,seq_len,d_model)
                k : torch.Tensor,
                v : torch.Tensor,
                #tile_size : int = 16,
                is_causal : bool = False):
        
        tile_size = 32
        if q.shape[1]%tile_size!=0:
            raise ValueError("tile size and given query shape not compatible")
        final_output = torch.empty_like(q)
        final_logsum_exp = torch.empty((q.shape[0],q.shape[1]))
        tile_q = torch.arange(0,q.shape[1],tile_size)
        d_model = q.shape[-1]
        
        for i in tile_q:
            current_q = q[:,i:i+tile_size,:] # (batch,tile_size,d_model)
            current_tile_size = current_q.shape[1]
            
            tile_output = torch.zeros_like(current_q) # (batch,tile_size,d_model)
            norm_factor = torch.zeros(q.shape[0],current_tile_size, 1, device=q.device) # (batch,tile_size,1)
            max_el = torch.full((q.shape[0],current_tile_size, 1),-torch.inf, device=q.device) # (batch,tile_size,1)
            
            for j in tile_q:
                current_k = k[:,j:j+tile_size,:] #(batch,tile_size,d_model)
                current_v = v[:,j:j+tile_size,:] #(batch,tile_size,d_model)
                
                mat_mul = (current_q@current_k.permute(0,2,1))/d_model**0.5 # (batch,tile_size,tile_size)
                row_max = torch.max(mat_mul,dim=-1, keepdim=True).values # (batch,tile_size,1)
                last_max_el = max_el
                max_el = torch.maximum(max_el,row_max) #(batch,tile_size) #we want to braodcast max_el over all elements in the tile
                numer = torch.exp(mat_mul - max_el)  # (batch,tile_size,tile_size)
                
                rescale_factor = torch.exp(last_max_el - max_el) # (batch,tile_size)
                norm_factor =  rescale_factor*norm_factor + torch.sum(numer,dim=-1, keepdim=True)
                tile_output = tile_output*rescale_factor + numer@current_v # reslolving previous tiles

            final_output[:,i:i+tile_size,:] = tile_output/norm_factor   # (batch,tile_size,d_model)
            final_logsum_exp[:,i:i+tile_size] = (max_el + torch.log(norm_factor)).squeeze(-1) #(batch,tile_size)
            
        ctx.save_for_backward(q, k, v, final_output, final_logsum_exp)
        return final_output
    
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError