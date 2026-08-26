import torch  
import math
from collections.abc import Iterable


class AdamW(torch.optim.Optimizer):
    def __init__ (self, params, lr=1e-3, eps = 1e-8,weight_decay =0.01 ,betas:tuple[float,float] = (0.9,0.999)):
        if lr < 0:
            raise ValueError (f"Invalid learning rate:{lr}")
        defaults =dict(lr =lr,betas =betas,eps= eps,weight_decay =weight_decay)
        super().__init__(params,defaults)

    
    @torch.no_grad() #进入step方法瞬间，全局强制关闭pytorch的梯度计算
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): #临时打破外层torch.no_grad的限制，重新开启AutoGrad梯度追踪
                loss = closure()

        #param_groups是defaults字典里创建的，此时提取当前组的超参数
        for group in self.param_groups:

            #提取当前一组参数专属的超参数
            beta1,beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            #用这一组专属的超参数，去更新这一组里所有参数p
            for p in group['params']:
                if p.grad is None:
                    continue
                grad= p.grad

                #获取当前参数p的状态字典
                state =self.state[p]

                #如果是第一次更新参数，初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)  #一阶矩
                    state['exp_avg_sq'] =torch.zeros_like(p)  #二阶矩


                state['step'] += 1
                t =state['step']
                m =state['exp_avg']
                v = state['exp_avg_sq']
                
                #一阶矩
                m.mul_(beta1).add_(grad,alpha=1 - beta1)

                #二阶矩
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                #偏差修正
                m_hat = m / (1- beta1 **t)
                v_hat = v / (1- beta2 **t)


                #解耦权重衰减
                p.mul_(1 - lr * weight_decay)

                #自适应梯度更新
                p.add_(m_hat/ (torch.sqrt(v_hat) + eps),alpha = -lr)


        return loss

def lr_cosine_schedule(it:int,max_learning_rate:float,min_learning_rate:float,warmup_iters:int,cosine_cycle_iters:int):
    #热身阶段,线性增长
    if it < warmup_iters:
        return   (it / warmup_iters) * max_learning_rate

    #保底阶段
    elif it > cosine_cycle_iters:
        return min_learning_rate
    
    #余弦退火阶段
    else:

        return min_learning_rate + 0.5 *(1+ math.cos(((it - warmup_iters)/(cosine_cycle_iters - warmup_iters))* math.pi)) * (max_learning_rate - min_learning_rate)




def gradient_clipping(parameters:Iterable[torch.nn.Parameter],max_l2_norm:float) -> None:
    #转为list，可以多次遍历
    params =list(parameters)
    total_norm_sq = 0.0 
    for p in params:
        if p.grad is None:
            continue
        total_norm_sq = total_norm_sq + (p.grad ** 2).sum().item()

    total_norm = total_norm_sq ** 0.5 


    #计算缩放系数
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef < 1.0 :
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
