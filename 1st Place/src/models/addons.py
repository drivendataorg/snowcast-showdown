import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.init as init
from collections import OrderedDict
import numpy as np
import random
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from threading import Thread
import os
 
def set_seed(seed, deterministic=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
        
def _applyfn(module, fn):
    for key in module.__dict__:
        if isinstance(module.__dict__[key], torch.Tensor):
            module.__dict__[key] = fn(module.__dict__[key])

class Linear2Function(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias, eps=None, mom=None, covx=None, step=1):
        # u = None
        if covx is not None and input.shape[0] > 1:
            nrm = input - input.mean(0, keepdim=True)
            covx.data.addmm_(nrm.t(), nrm, beta=mom, alpha=(1.-mom)/input.shape[0])
            # covx.data.addmm_(input.t(), input, beta=mom, alpha=(1.-mom)/input.shape[0])
            # u = covx/(1.-mom**step)
            ctx.eps=eps; ctx.mom=mom; ctx.step=step
        ctx.save_for_backward(input, weight, covx)
        ctx.withbias = bias is not None
        return input.mm(weight.t()) + bias if ctx.withbias else input.mm(weight.t())
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, covx = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)            
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if covx is not None and input.shape[0] > 1:
                # grad_weight.add_(grad_weight.mean(1,keepdim=True), alpha=-0.9)
                # grad_weight.sub_(grad_weight.mean(1,keepdim=True))
                # grad_weight.div_(torch.sqrt((grad_weight*grad_weight).mean(1,keepdim=True)).add_(eps))
                # grad_weight.div_(torch.abs(grad_weight).mean(1,keepdim=True).add_(eps))
                u = covx/(1.-ctx.mom**ctx.step)
                d = torch.sqrt(u.diagonal())
                eye = torch.eye(*u.shape, out=torch.empty_like(u))
                (u.div_(d[None,:]).div_(d[:,None])).add_(eye, alpha=1.+ctx.eps)
                fp16 = u.dtype == torch.float16
                # grad_weight = torch.mm(grad_weight, u.inverse())
                gw = (grad_weight.float() if fp16 else grad_weight).unsqueeze(-1)
                if hasattr(torch, 'linalg'):
                    grad_weight = torch.linalg.solve(u.float() if fp16 else u,gw).squeeze(-1)
                else:
                    grad_weight = torch.solve(gw,u.float() if fp16 else u)[0].squeeze(-1)
                if fp16:
                    grad_weight = grad_weight.half()
                # grad_weight.div_(torch.sqrt((grad_weight*grad_weight).mean(1,keepdim=True)).add_(eps))
                # grad_weight = torch.mm(grad_weight, u.float().inverse().to(grad_weight.dtype))    
        if ctx.withbias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None
class Linear2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=1e-6, mom=0.9):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)        
        self.eps,self.mom = eps, torch.autograd.Variable(torch.tensor(mom), requires_grad=False)
        self.reset_parameters ()
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        self.covx = torch.autograd.Variable(torch.zeros(self.weight.shape[1],self.weight.shape[1],dtype=self.weight.dtype,device=self.weight.device), requires_grad=False)
        self.step = 0
    def forward(self, input):
        sh = list(input.shape)
        if self.training:
            if self.covx.device == input.device:
                self.step += 1
                step = self.step
                covx = self.covx
            else:
                step = self.step + 1
                covx = self.covx.to(input.device)
            if len(sh) > 2:
                return Linear2Function.apply(input.reshape(-1,sh[-1]), self.weight, self.bias, self.eps, self.mom, covx, step).reshape(sh[:-1]+[-1])
            else:
                return Linear2Function.apply(input, self.weight, self.bias, self.eps, self.mom, covx, step)
        else:
            if len(sh) > 2:
                return Linear2Function.apply(input.reshape(-1,sh[-1]), self.weight, self.bias).reshape(sh[:-1]+[-1])
            else:
                return Linear2Function.apply(input, self.weight, self.bias)
    def _apply(self, fn):
        nn.Linear._apply(self, fn)
        _applyfn(self, fn)
        return self
class LinearConv2(Linear2):
    def __init__(self, in_features, out_features, bias=True, eps=1e-6, mom=0.9):
        Linear2.__init__(self, in_features, out_features, bias=bias, eps=eps, mom=mom)
    def forward(self, x):
        return Linear2.forward(self, x.permute(0,2,1)).permute(0,2,1)
        
class normalizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, dims=[0], momentum=0.1, eps=1e-5, running_mean=None, running_var=None, factor=None):
        if factor is not None:
            mean = x.mean(dims, keepdim=True).float()
            if torch.isfinite(mean).all():
                running_mean.data.mul_(1. - momentum).add_(mean,alpha=momentum)
                running_var.data.mul_(1. - momentum).add_((x*x).mean(dims, keepdim=True).float(),alpha=momentum)
                # xmin = x.min(dims[-1], keepdim=True)[0]
                # xmax = x.max(dims[-1], keepdim=True)[0]
                # for d in dims[:-1]:
                #     xmin = xmin.min(d, keepdim=True)[0]
                #     xmax = xmax.max(d, keepdim=True)[0]
                # running_var.data.mul_(1. - momentum).add_((xmax-xmin).float(),alpha=momentum)
                factor.data.mul_(1. - momentum).add_(momentum)
                bias1 = -running_mean/factor
                weight.data.copy_((1./(torch.sqrt((running_var/factor - bias1*bias1 + eps).clamp_(min=eps)))).to(x.dtype))
                # weight.data.copy_((1./(running_var/factor+eps)).to(x.dtype))
                bias.data.copy_((weight*bias1).to(x.dtype))
        ctx.save_for_backward(weight)
        if x.is_mkldnn:
            return (x.to_dense() * weight + bias).to_mkldnn()
        else:
            return x * weight + bias
    @staticmethod
    def backward(ctx, grad_output):
        grad = None
        if ctx.needs_input_grad[0]:
            grad = ctx.saved_tensors[0]*grad_output
        return grad, None, None, None, None, None, None, None, None
    
class Normalize(nn.Module):
    def __init__(self, num_features, dims=[0], eps=1e-5, momentum=0.01, frozen=False, mkl=False, *args, **kwargs):
        super(Normalize, self).__init__()        
        self.num_features = num_features
        self.dims = dims
        self.eps = eps
        self.mkl = mkl
        self.momentum = momentum
        self.frozen = frozen
        shape = [1]*(len(dims)+len(num_features))
        for j,i in enumerate ([i for i in range(len(shape)) if i not in dims]):
            shape[i] = num_features[j]
        self.register_buffer('weight', torch.ones(*shape))
        self.register_buffer('bias', torch.zeros(*shape))
        self.register_buffer('running_mean', torch.zeros(*shape))
        self.register_buffer('running_var', torch.ones(*shape))
        self.register_buffer('factor', torch.zeros(1))
    def _apply(self, fn):
        nn.Module._apply(self, fn)
        _applyfn(self, fn)
        return self
    def float(self):
        fp16 = self.weight.dtype is torch.float16
        nn.Module.float(self)
        if fp16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()

    def forward(self, x: torch.Tensor):
        if not self.frozen and self.training and self.momentum > 0:
            y = normalizeFunc.apply(x, self.weight, self.bias, self.dims, self.momentum, self.eps, self.running_mean, self.running_var, self.factor)
        else:
            y = normalizeFunc.apply(x, self.weight, self.bias)
        return y.to_mkldnn() if self.mkl else y
            
    # def frombn(self,bn):
    #     self.factor = torch.ones(1)
    #     with torch.no_grad():
    #         self.running_var = bn.running_var/bn.weight.data/bn.weight.data - self.eps
    #         self.weight = bn.weight.data/torch.sqrt(bn.running_var + self.eps)
    #         self.bias   = bn.bias.data - bn.running_mean * self.weight
    #         self.running_mean = bn.running_mean - bn.bias.data / self.weight
    #         self.running_var += self.running_mean*self.running_mean
        
def weight_init(m,gain=1.):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if hasattr(m,'bias'):    
        if m.bias is not None and isinstance(m.bias, torch.Tensor):
            # init.constant_(m.bias.data, 0)
            init.normal_(m.bias.data, mean=0, std=0.1*gain)
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.Conv3d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.ConvTranspose1d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02*gain)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02*gain)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02*gain)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data,gain=gain)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, std=gain*0.1, mean=param.data.mean())
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, std=gain)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, std=gain)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, std=gain)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_(-0.01,0.01)

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0., L1=0.,
                       warmup = 100, amsgrad=False, belief=False, gc=0.9, parallel=True, gradinit=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gc <= 1.0:
            raise ValueError("Invalid gc value: {}".format(gc))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup, 
                        amsgrad=amsgrad, belief=belief, gc=gc, gradinit=gradinit, L1=L1)
        params = list(params)
        super(AdamW, self).__init__([params[i] for i in np.argsort([p.numel() for p in params])[-1::-1]], defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            
    def stepone(self, group, p):
        p_data_fp32 = p.data.float()
        grad = p.grad.data.float()
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p_data_fp32)
            state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
            if group['amsgrad']:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # else:
        #     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
        #     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
        
        warmup = group['warmup'] > state['step']
        gradinit = group['gradinit'] and warmup and p_data_fp32.ndim>1 and p_data_fp32.numel() > 8
        if gradinit:
            # grad = torch.full_like(p_data_fp32, (grad*p_data_fp32).mean())
            grad = p_data_fp32*(grad*p_data_fp32).mean(0,keepdim=True)
            # grad = (grad*p_data_fp32).mean(0,keepdim=True).expand_as(p_data_fp32)
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1. - beta1 ** state['step']
        bias_correction2 = math.sqrt(1. - beta2 ** state['step'])
                
        # Gradient centralization
        if group['gc']>0. and grad.ndim > 1:
            grad.add_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True), alpha=-group['gc'])
        if group['belief'] and grad.ndim > 1:
            grad_residual = grad.add(exp_avg, alpha=-1./bias_correction1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1. - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1. - beta2)
        exp_avg.mul_(beta1).add_(grad, alpha=1. - beta1)
                
        scheduled_lr = 1e-8+state['step']*group['lr']/group['warmup'] if warmup and not gradinit else group['lr']
        if group['weight_decay'] != 0:
            p_data_fp32.mul_(1. - group['weight_decay']*scheduled_lr)
        if group['L1'] != 0:
            p_data_fp32.add_(-p_data_fp32.sign() * group['L1']*scheduled_lr)
        if group['amsgrad']:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(state['max_exp_avg_sq']*beta2, exp_avg_sq, out=state['max_exp_avg_sq'])
            denom = state['max_exp_avg_sq'].sqrt()
        else:
            denom = exp_avg_sq.sqrt()
        p_data_fp32.addcdiv_(exp_avg, denom.add_(group['eps']*bias_correction2), 
                             value = -scheduled_lr*bias_correction2/bias_correction1)
        p.data.copy_(p_data_fp32)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        threads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                if torch.isnan(grad).any() or (grad == 0.).all():
                    continue
                if sum([d>1 for d in p.data.shape])>1:
                    threads.append(Thread (target=self.stepone, args=(group, p)))
                    threads[-1].start()
                else:
                    self.stepone(group, p)
        for thread in threads:
            thread.join()
        return loss

def reduce_mem_usage(df, use_float16=False,output=print):
#    """
#    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
#    """
#    start_mem = df.memory_usage().sum() / 1024**2
#    output("Memory usage of dataframe is {:.2f} MB".format(start_mem))    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
#    end_mem = df.memory_usage().sum() / 1024**2
#    output("Memory usage after optimization is: {:.2f} MB".format(end_mem))
#    output("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))    
    return df                