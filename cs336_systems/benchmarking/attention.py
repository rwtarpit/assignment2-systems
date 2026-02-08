import torch
import torch.nn as nn
import timeit
from itertools import product
from einops import rearrange, einsum, reduce
#torch._functorch.config.donated_buffer = False


def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    max_el = torch.max(x, dim=dim, keepdim=True).values
    x = x-max_el
    num = torch.exp(x)
    denom = torch.sum(num, dim=dim, keepdim=True)
    return num/denom

def self_attention_(q : torch.Tensor, # batch x seq_len x d_model
                   k : torch.Tensor,
                   v : torch.Tensor,
                   mask = None):     # seq_Len x seq_len
    d_k = k.shape[-1]
    scores = (q@k.transpose(-2,-1))/(d_k**0.5)
    if mask is not None:
        scores = torch.where(mask.bool(), scores, float("-inf"))
        
    return softmax(scores, dim=-1)@v


def benchmark_attention(compiled: bool = False):
    batch_size = 8
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_iters = 100
    
    print("Executing on:", device)
    
    # Handle Global vs Local scope for the compiled function
    if compiled:
        # Use a local reference to avoid global scope confusion
        current_attn = torch.compile(self_attention_)
        print("Compiling naive attention with torch inductor")
    else:
        current_attn = self_attention_
        print("Benchmarking naive attention (Eager)")

    print(f"{'D_model':>8} | {'Seq_Len':>8} | {'Fwd Sum(s)':>12} | {'Bwd Sum(s)':>12} | {'Peak Mem (MB)':>15}")
    print("-" * 75)
    
    for d_model, seq_len in product(d_model_list, seq_len_list):
        try:
            # random data 
            q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

            # --- Warmup ---
            for _ in range(10):
                out = current_attn(q, k, v, mask)
                out.sum().backward()
            torch.cuda.synchronize()

            # --- Benchmark Forward Passes ---
            fwd_total_time = 0.0
            for _ in range(num_iters):
                torch.cuda.synchronize()
                start = timeit.default_timer()
                _ = current_attn(q, k, v, mask)
                torch.cuda.synchronize()
                fwd_total_time += (timeit.default_timer() - start)

            # --- Measure Memory (Activations) ---
            torch.cuda.reset_peak_memory_stats(device)
            out = current_attn(q, k, v, mask)
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # --- Benchmark Backward Passes ---
            bwd_total_time = 0.0
            for _ in range(num_iters):
                out_for_bwd = current_attn(q, k, v, mask)
                loss = out_for_bwd.sum()
                
                if q.grad is not None: q.grad.zero_()
                if k.grad is not None: k.grad.zero_()
                if v.grad is not None: v.grad.zero_()
                
                torch.cuda.synchronize()
                start = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                bwd_total_time += (timeit.default_timer() - start)

            print(f"{d_model:8d} | {seq_len:8d} | {fwd_total_time:12.4f} | {bwd_total_time:12.4f} | {peak_mem_mb:15.2f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{d_model:8d} | {seq_len:8d} | {'OOM':>12} | {'OOM':>12} | {'OOM':>15}")
                torch.cuda.empty_cache()
            else:
                raise e

if __name__=="__main__":
    benchmark_attention(compiled = True)
    benchmark_attention(compiled = False)