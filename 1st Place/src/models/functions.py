# import math
import torch
import torch.nn as nn
import numpy as np
        
class softdot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, b, dim=0, count=None):        
        ctx.dim = dim
        ctx.nocount = count is None
        if ctx.nocount:
            dist = nn.functional.softmax(d.data,dim=dim)
            ctx.save_for_backward(b,dist)
        else:
            dist = torch.exp(d.data)
            sdist = dist.sum(dim=dim,keepdim=True)
            # print([[(sdist/count<q).float().mean().item()] for q in [0.75,1.,1.25,1.5]])
            limited = sdist<count
            sdist[limited] = count[limited].to(dist.dtype) if type(count) is torch.tensor else count
            dist.div_(sdist)
            ctx.save_for_backward(b,dist,~limited)
        if dist.ndim < b.ndim:
            dist = torch.unsqueeze(dist,b.ndim-1)
        return (b.data*dist).sum(dim=dim).detach()
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            b = ctx.saved_tensors[0]; dist = ctx.saved_tensors[1]
            grad = b.data*grad_output.unsqueeze(ctx.dim)
            if dist.ndim < b.ndim:
                grad = grad.sum(-1)
            grad.mul_(dist.data)
            if ctx.nocount:                
                grad.sub_(dist.data*(grad.sum(dim=ctx.dim,keepdims=True)))
            else:
                grad.sub_(dist.data*grad.sum(dim=ctx.dim,keepdims=True)*ctx.saved_tensors[2].float())
            return grad,None,None,None
        else:
            return None,None,None,None

class softdot1(softdot):
    @staticmethod
    def backward(ctx, grad_output):
        grad=softdot.backward(ctx, grad_output)[0] if ctx.needs_input_grad[0] else None
        gradb = None
        if ctx.needs_input_grad[1]:
            dist = ctx.saved_tensors[1]
            if dist.ndim < ctx.saved_tensors[0].ndim:
                gradb = dist.data[:,:,None]*torch.unsqueeze(grad_output,ctx.dim)
            else:
                gradb = dist.data*torch.unsqueeze(grad_output,ctx.dim)
        return grad,gradb,None,None     

class xp1expm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exp):        
        ctx.exp = exp     
        y = torch.sqrt(x.data)
        if ctx.exp:
            z = torch.exp(-y)
            ctx.save_for_backward(z)
            return ((y+1)*z).detach()
        else:
            ctx.save_for_backward(y)
            return (torch.log1p(y).sub_(y)).detach()
    @staticmethod
    def backward(ctx, grad_output):        
        if ctx.needs_input_grad[0]:
            y = ctx.saved_tensors[0]
            return (grad_output*y if ctx.exp else grad_output/(y+1)).mul_(-0.5), None
        return None, None
        
class sigma_act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x.clamp(min=0.) + 1./(1. - x.clamp(max=0.))).detach()
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            x1 = ctx.saved_tensors[0].clamp(max=0.)-1.
            return grad_output/x1/x1
        return None
