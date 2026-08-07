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


class RMSNorm(nn.Module):
    def __init__(self,d_model: int ,eps:float =1e-5,device= None,dtype= None):
        super().__init__()
        self.eps = eps #ps保存self，供forward使用
        r = torch.ones(d_model,device =device,dtype = dtype)
        self.weight = nn.Parameter(r)


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        in_dtype = x.dtype #记录输入数据类型
        x_fp32 = x.to(torch.float32) #数值稳定的上采样
        ms =x_fp32.pow(2).mean(dim=-1,keepdim=True) #沿最后维度计算均方值，且保留纬度用于广播
        rsqrt_rms = torch.rsqrt(ms + self.eps) #计算均方值
        normed = x_fp32 * rsqrt_rms * self.weight #归一化2
        x=normed.to(in_dtype)
        return x 

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff: int | None = None,device = None,dtype =None):
        super().__init__()
        if d_ff is None:
            d_ff =64* ((int(8 * d_model /3) + 63) // 64)#向上取整
        #特征提取
        self.w1 =Linear(in_features = d_model,out_features =d_ff,device = device ,dtype = dtype)
        
        #汇总输出
        self.w2 =Linear(in_features = d_ff,out_features = d_model,device = device,dtype =dtype)
        
        #流量控制器
        self.w3 =Linear(in_features = d_model,out_features=d_ff,device= device,dtype =dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #主特征张量
        out1 = self.w1(x)
        #门控信号张量
        out3 = self.w3(x)

        return self.w2(out1 * torch.sigmoid(out1) * out3)