import torch
import torch.nn as nn
import math

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

#均方根归一化
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

#SwiGLU前馈神经网络
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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k: int,max_seq_len: int,device=None):
        super().__init__()

        #i是一个类似list的tensor
        i = torch.arange(0,d_k,2,device=device,dtype=torch.float32)
        freq =   1.0/(theta **(i/d_k) )

        m =torch.arange(0,max_seq_len,1,device =device,dtype=torch.float32)
        #m 和 freq相互点乘
        angles = torch.outer(m,freq)

        
        cos_emb = torch.cos(angles)
        sin_emb = torch.sin(angles)
        
        cos_emb =cos_emb.repeat_interleave(2,dim =-1)
        sin_emb =sin_emb.repeat_interleave(2,dim =-1)
        self.register_buffer("cos_cached",cos_emb,persistent =False)
        self.register_buffer("sin_cached",sin_emb,persistent =False)
     
     
    def forward(self,x:torch.Tensor,token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x1 =x[...,0::2]
        x2 =x[...,1::2]

        #交错配对
        x_stacked = torch.stack([-x2,x1],dim =-1)

        #展平恢复
        x_tilde = x_stacked.flatten(start_dim = -2)

        return  x * cos + x_tilde * sin

def softmax(x: torch.Tensor,dim: int)-> torch.Tensor:

    #取最大值
    m = torch.max(x,dim =dim,keepdim =True).values
    exp_x = torch.exp(x -m)
    sum_exp = torch.sum(exp_x,dim = dim,keepdim = True)

    return exp_x /sum_exp


def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor,v: torch.Tensor,mask: torch.Tensor |  None = None):
    d_k =q.shape[-1]
    #转置k
    k_t = k.transpose(-2,-1)
    score =q @ k_t

    score = score / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(mask == False, float("-inf"))


    attn_weight = softmax(score,dim =-1)
    output = attn_weight @ v
    return output


class Causal_multi_head_self_attention(nn.Module):
    def __init__(self,d_model: int ,num_heads: int,device = None,dtype =None,max_seq_len:int | None = None,theta:float | None = None):
        super().__init__()



        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model //num_heads #计算单个head的特征维度

        #定义旋转位置编码
        if theta and max_seq_len:
            self.rope = RotaryPositionalEmbedding(theta=theta,d_k = self.d_k,max_seq_len = max_seq_len,device=device)
        else:
            self.rope = None
        #WQ
        self.q_proj = Linear(in_features=d_model,out_features= d_model,device = device,dtype =dtype)
        #Wk
        self.k_proj = Linear(in_features=d_model,out_features= d_model,device = device,dtype =dtype)
        #Wv
        self.v_proj = Linear(in_features=d_model,out_features= d_model,device = device,dtype =dtype)
        self.output_proj = Linear(in_features=d_model,out_features= d_model,device = device,dtype =dtype)

    def forward(self,x :torch.Tensor, token_positions: torch.Tensor | None = None):
        batch_size ,seq_len,_ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        #把d_model的维度，切分成两个子维度(num_heads,d_k)
        q = q.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        k = k.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        v = v.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len,device = x.device)

            #对q和k分别施加RoPe融合
            q = self.rope(q,token_positions)           
            k = self.rope(k,token_positions)

        #tril（triangular lower)只保留矩阵的下三角
        causal_mask = torch.tril(torch.ones((seq_len,seq_len),device = x.device,dtype= torch.bool))

        attn_out = scaled_dot_product_attention(q,k,v,mask=causal_mask)
        

        #contiguous在显存里真正把数据重新拷贝为连续的内存块，再将(num_heads,d_k)重新合成一个单维度d_model
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        return self.output_proj(attn_out)

        

class TransformerBlock(nn.Module):
    def __init__(self,d_model: int,num_heads:int,d_ff:int,max_seq_len:int| None = None,theta:float |None =None,eps:float =1e-5,device =None,dtype =None):
        super().__init__()
        
        #均方根归一化
        self.ln1 = RMSNorm(d_model =d_model,eps= eps,device=device,dtype=dtype)
        #注意力层
        self.attn = Causal_multi_head_self_attention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta,device=device,dtype=dtype)
        self.ln2=RMSNorm(d_model =d_model,eps= eps,device=device,dtype=dtype)
        #SWiGLu前馈函数
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor |None = None)-> torch.Tensor:
        #注意力残差
        x= x +self.attn(self.ln1(x),token_positions=token_positions)


        #前馈网络残差
        x = x + self.ffn(self.ln2(x))
        return x 

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int, context_length:int,d_model: int,num_layers: int,num_heads:int,d_ff: int,rope_theta:float,eps:float =1e-5,device= None,dtype=None):
        super().__init__()
        self.context_length =context_length
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        #未来张量可以传进来进行num_layers的计算
        self.layers =nn.ModuleList([
            TransformerBlock(d_model = d_model,num_heads = num_heads,d_ff = d_ff,max_seq_len=context_length,theta = rope_theta,eps = eps,device=device ,dtype=dtype)
            for _ in range(num_layers)
        ]) 
        self.ln_final = RMSNorm(d_model = d_model,eps = eps ,device=device,dtype =dtype)
        self.lm_head = Linear(in_features = d_model,out_features=vocab_size,device= device,dtype =dtype)


    def forward(self,in_indices:torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits 

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    #取最大值
    max_value =torch.max(logits,dim=-1,keepdim=True).values
    log_sum_exp = max_value.squeeze(-1) + torch.log(torch.sum(torch.exp(logits-max_value),dim = -1))

    target_logits = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    
    losses = log_sum_exp - target_logits

    return losses.mean()


def sample_decoding(model:TransformerLM,max_new_tokens:int,eos_token_id:int, prompt_indices:torch.Tensor,temperature=1.0,top_p = 1.0):
    model.eval()
    #禁用梯度计算
    with torch.no_grad():
        #初始化动态序列变量
        current_ids = prompt_indices.clone()

        for _ in range(max_new_tokens):
            #第一个维度: 保留所有的batch，第二个-model.context_length:，从倒数第model.context_length 个元素一直取到最末尾
            #取末尾是因为离它最近的上下文比很久以前的重要
            current_input = current_ids[:, -model.context_length:]
            logits = model(current_input)
            #logits 形状：(batch_size, sequence_length, vocab_size)（三维张量）
            next_token_logits = logits[:, -1, :]

            #temperature 是改变 Logits 相对差距、调控随机性的极佳手段
            next_token_logits = next_token_logits / temperature 

            #按得分从大到小排序
            sorted_logits ,sorted_indices = torch.sort(next_token_logits,descending = True,dim = -1)
            # 计算softmax的排序后的概率
            sorted_probs = torch.softmax(sorted_logits ,dim = -1)
            
            # 计算softmax的累计概率，累积求和/前缀求和
            cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

            #找出累计概率超过top_p 的位置
            sorted_indices_to_remove = cumulative_probs > top_p #里面存的bool

            #将掩码右移一位，确保累积概率恰好刚好达到top_p的那个临界词也被保留下来
            sorted_indices_to_remove[:,1:] = sorted_indices_to_remove[:,:-1].clone()
            
            #把概率最高的设定为false，保护概率最大的那位
            sorted_indices_to_remove[: ,0] = False

            #把true全部设置为inf
            sorted_logits [sorted_indices_to_remove] = -float('inf')
            
            #归一化概率
            probs = torch.softmax(sorted_logits,dim = -1 )

            #概率抽样
            next_sorted_token = torch.multinomial(probs,num_samples = 1)

            #查追溯地图找回原Token ID
            next_token = torch.gather(sorted_indices ,dim= -1,index =next_sorted_token)


            #拼接到当前序列中
            current_ids = torch.cat([current_ids,next_token],dim =-1)

            if (next_token == eos_token_id).all():
                break

        return current_ids

            
