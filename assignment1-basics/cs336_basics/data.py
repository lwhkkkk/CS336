import torch
import numpy as np

def get_batch(x:np.ndarray,batch_size:int,context_length:int,device:str):
    #batch起始索引
    indices =  np.random.randint(0, len(x)-context_length  ,size = batch_size)

    
    inputs = np.array([x[i : i+context_length] for i in indices])
    targets = np.array([x[i+ 1: i+ context_length + 1]for i in indices])

    inputs = torch.tensor(inputs,dtype=torch.long , device=device)
    targets = torch.tensor(targets,dtype = torch.long ,device = device)

    return inputs,targets