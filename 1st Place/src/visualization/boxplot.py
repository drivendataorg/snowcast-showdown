import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def quantized (x,y,levels=6,nrmy=True,nrmx=False):
    if nrmy:
        nrm = torch.nan_to_num(y,0.)
        nrm = nrm.sum(1)/(nrm>0).float().sum(1)
        y = y/nrm[:,None]
        if nrmx:
            x = x/nrm[:,None]*10
        ok = torch.isfinite(x+y)&(y>0)&(y<5)
    else:
        ok = torch.isfinite(x+y)
    x  = x[ok].cpu().numpy(); y=y[ok].cpu().numpy()
    s  = np.argsort(x); x=x[s]
    x0 = x[0]; x1=x[-1]; x=(x-x0)*(levels/(x1-x0))
    xq = np.arange(x.shape[0])*levels//x.shape[0]
    for i in range(levels):
        x[xq==i] = np.mean(x[xq==i])
    x = x*((x1-x0)/levels)+x0
    dec = int(np.floor(3-np.log10(x1-x0)))
    return np.round(x,dec),y[s]

def boxplot (inputs,arglist):
    for i,a in enumerate(arglist):
        print({a: {mode: (torch.isfinite(inputs[mode]['xinput'][...,i]).sum().item(),
                          torch.isfinite(inputs[mode]['yinput'][...,i]).sum().item()) for mode in inputs}})
        fig, ax = plt.subplots(1,1)
        ax.set_title(a)
        for mode in ['train']:
            try:
                xi,xt = quantized (inputs[mode]['xinput'][...,i], inputs[mode]['xval'][...,0],nrmx=a in ['sde','sdwe'])
                yi,yt = quantized (inputs[mode]['yinput'][...,i], inputs[mode]['yval'][...,0],nrmx=a in ['sde','sdwe'])
                sns.boxenplot(x=xi,y=xt, ax=ax, linewidth=1.5)
                sns.boxenplot(x=yi,y=yt, ax=ax, linewidth=1.5)
                # ax.plot(xi,xt, '.'); ax.plot(yi,yt, '.')
            except:
                pass