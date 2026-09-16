import timeit
import argparse
import numpy as np
import torch
import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM 
from torch.optim import AdamW

def parse_args():
    parser = argparse.ArgumentParser(description="描述信息")

    #测试模型与步数控制
    parser.add_argument("--mode", type = str ,choices=["forward" , "forward_backward","full"], default ="forward", help ="Benchmarking mode")
    parser.add_argument("--warmup_steps" , type =int, default=5, help= "Number of warmup steps")
    parser.add_argument("--num_steps", type = int,default= 10,help="Number of measurement step")

    #数据与硬件参数
    parser.add_argument("--batch_size",type =int,default=4)
    parser.add_argument("--context_length",type =int,default =512)
    parser.add_argument("--device",type =str,default = "cuda")
    
    parser.add_argument("--vocab_size",type = int,default = 10000)
    parser.add_argument("--d_model", type = int ,default =512)
    parser.add_argument("--num_layers", type = int ,default = 4)
    parser.add_argument("--num_heads", type = int, default =8)
    parser.add_argument("--d_ff", type = int,default =2048)

    return parser.parse_args()

#单步执行操作 
def run_step(input, model, mode, optimizer =None ):
    if mode in ["forward_backward", "full"]:
        model.zero_grad()

    logits = model(input)
    loss = logits.sum()
    
    if mode in ["forward_backward", "full"]:
        loss.backward()
        if mode == "full" and optimizer is not None:
            optimizer.step()
    
    torch.cuda.synchronize()

    




def main():
    args = parse_args()
    model = BasicsTransformerLM(
        vocab_size =args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads =args.num_heads,
        d_ff = args.d_ff

    )
    model.to(args.device)
    #构造随机整数张量(high: 随机整数的上限/词表的大小 size: 包含维度大小的tuple，形状大小and序列长度构成  )
    inputs = torch.randint(high = args.vocab_size,size = (args.batch_size,args.context_length) ,device=args.device)
    
    optimizer = AdamW(model.parameters(), lr= 1e-3) if args.mode =="full" else None

    #预热步数
    for _ in range(args.warmup_steps):
        run_step(inputs, model, args.mode, optimizer)


    #正式测量循环
    times = []
    for _ in range(args.num_steps):
        start_time = timeit.default_timer()
        run_step(inputs, model, args.mode, optimizer)
        end_time = timeit.default_timer()

        times.append(end_time -start_time)

    #平均耗时
    mean_time = np.mean(times)

    #标准差
    std_time = np.std(times)
    print(f"mode: {args.mode}")
    print(f"Mean time: {mean_time * 1000:.3f} ms")
    print(f"Std time: {std_time * 1000:.3f} ms")
    print(f"Model config: d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}")

if __name__ == "__main__":
    main()
    

    