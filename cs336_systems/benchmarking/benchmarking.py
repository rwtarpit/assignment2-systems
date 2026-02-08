import torch
import sys
import os
import timeit
import argparse
from statistics import mean, stdev

from cs336_basics.model import BasicsTransformerLM

def benchmark_time(model : torch.nn.Module,
                   x : torch.Tensor,
                   y : torch.Tensor):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    

    def one_pass():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
    for _ in range(warmup_steps):
        one_pass()
        
    times = []
    for _ in range(profiling_steps):
        start_time = timeit.default_timer()
        one_pass()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
        
    return mean(times), stdev(times)

    
def parse_args():
    parser = argparse.ArgumentParser(description="speed profiling of LLMs")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    #parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    #parser.add_argument("--checkpoint", type=str, default="benchmark_checkpoint.json", help="Checkpoint file path")
    #parser.add_argument('--mixed_precision', type=bool, default=False, help='Use mixed precision to run the model')
    return parser.parse_args()

# HyperParams
model_config = {"d_model": 512, "d_ff": 1344, "num_layers": 12, "num_heads": 16}
vocab_size = 10_000
context_length = 256
batch_size = 1
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    
    global context_length
    if args.context_length:
        context_length = args.context_length
        
    global warmup_steps
    if args.warmup_steps:
        warmup_steps = args.warmup_steps
        
    if args.all:
        run_config = model_config
    elif args.d_model and args.d_ff and args.num_layers and args.num_heads:
        run_config = {
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }
    else:
        raise ValueError("Must specify either --all or all custom model hyperparameters.")
    
    