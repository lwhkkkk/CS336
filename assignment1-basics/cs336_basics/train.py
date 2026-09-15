import argparse
from cs336_basics.data import get_batch
import numpy as np
from cs336_basics.model import TransformerLM,cross_entropy
from cs336_basics.optimizer import AdamW
import torch
import os
import time 
import wandb








def parse_args():
    #数据与训练配置
    parser = argparse.ArgumentParser(description = "Train a Transformer LM")
    parser.add_argument("--train_data",type= str,required = True,help="默认数据路径")
    parser.add_argument("--val_data",type = str ,required = True,help= "验证集路径")
    parser.add_argument("--batch_size",type= int ,default = 16,help= "训练的batch")
    parser.add_argument("--context_length",type = int ,default= 256,help="文本长度")
    parser.add_argument("--max_steps",type = int,default= 10,help= "默认总训练步数")
    parser.add_argument("--device",type = str ,default="cpu")

    #模型超参数
    parser.add_argument("--vocab_size",type =int,default =10000,help="词表大小")
    parser.add_argument("--d_model",type = int,default = 512,help="模型维度")
    parser.add_argument("--num_layers" ,type = int,default = 6,help= "模型层次")
    parser.add_argument("--d_ff",type = int,default = 2048,help = "前馈网络FFN的隐藏层维度")
    parser.add_argument("--attn_pdrop",type = float,default = 0.1 ,help = "注意力Dropout丢弃概率")
    parser.add_argument("--num_heads", type = int ,default =8,help =" 多头注意力头数")
    parser.add_argument("--rope_theta",type = float,default =10000.0 ,help = "旋转位置编码的基数")

    #优化器与调度
    parser.add_argument("--lr",type = float,default = 5e-4,help ="最大学习率")
    parser.add_argument("--weight_decay",type = float,default = 0.1,help= "权重衰减率")
    parser.add_argument("--warmup_steps",type = int,default = 100, help= "预热步数")
    parser.add_argument("--grad_clip",type = float,default = 1.0,help="梯度裁剪的MaxL2范数阈值 ")
    parser.add_argument("--min_lr",type = float,default = 1e-5,help="余弦退火最低学习率")

    #记录与保存
    parser.add_argument("--eval_interval",type =int,default =100,help ="验证集评估的步数间隔")
    parser.add_argument("--save_interval",type = int,default = 500,help ="保存checkpoint的步数间隔")
    parser.add_argument("--checkpoint_dir",type =str,default ="checkpoints",help= "检查点文件保存目录")
    return parser.parse_args()













if __name__ == "__main__":
    args = parse_args()

    #初始化wandb
    wandb.init(
        project = "cs336-assignment1",
        name = f"transformer_lm_batch128{args.lr}",
        config = vars(args)
    )

    #记录起点
    start_time = time.time()


    train_data = np.memmap(args.train_data,dtype = np.uint16,mode ='r')
    val_data = np.memmap(args.val_data,dtype = np.uint16,mode= 'r')
    model = TransformerLM(vocab_size = args.vocab_size,
                        context_length = args.context_length,
                        d_model = args.d_model,
                        num_layers =args.num_layers,
                        num_heads = args.num_heads ,
                        d_ff = args.d_ff,
                        rope_theta= args.rope_theta,
                        device = args.device)

    optimizer = AdamW(model.parameters(),lr = args.lr,weight_decay =args.weight_decay)

    for step in range(1,args.max_steps + 1):

        #获取一个batch的数据
        inputs ,targets = get_batch(x = train_data, batch_size =args.batch_size,context_length =args.context_length,device =args.device)
        #把包含TokenID 的矩阵input 传给Transformer模型
        logits = model(inputs)

        #对比模型的预测输出logits和真实目标targets，计算出当前的标量损失值loss
        loss =cross_entropy(logits,targets)

        elapsed_time = time.time() -start_time

        wandb.log({
            "train/loss": loss.item(),
            "wall_clock_time" : elapsed_time,
        },step = step
        )

        #清空历史梯度的对象是优化器自身
        optimizer.zero_grad()

        #反向传播
        loss.backward()

        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)

        #优化器更新模型参数
        optimizer.step()

        print(f"Step {step} | train Loss:{loss.item():.4f}")  

        if step % args.eval_interval  == 0:
            model.eval()
            with torch.no_grad():
                
                val_inputs,val_targets =get_batch(x= val_data, batch_size = args.batch_size,context_length =args.context_length,device=args.device)
                #前向传播
                val_logits = model(val_inputs)
                #计算liss
                val_loss = cross_entropy(val_logits, val_targets) 

                wandb.log({
                    "val/loss" : val_loss.item(),
                }, step =step)

                print(f"Step {step} | val Loss:{val_loss.item():.4f}") 


        model.train()      


        if step % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir,exist_ok =True)
            checkpoint_dict = {}
            checkpoint_dict["model"] = model.state_dict()
            checkpoint_dict["optimizer"] =optimizer.state_dict()
            checkpoint_dict["step"] =step
            checkpoint_path =os.path.join(args.checkpoint_dir ,f"checkpoint_{step}.pt")
            torch.save(checkpoint_dict,checkpoint_path)

    wandb.finish()