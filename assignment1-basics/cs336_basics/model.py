import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        w = torch.empty(size=(out_features,in_features),device=device,dtype=dtype)
        torch.nn.init.trunc_normal_(w)
        self.weight =nn.Parameter(w)


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        e = torch.empty(size=(num_embeddings,embedding_dim),device=device,dtype=dtype)
        torch.nn.init.trunc_normal_(e)
        self.weight = nn.Parameter(e)

    def forward(self,token_ids:torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]
