import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from visualization.boxplot import quantized
from models.inference import getests

def monthests (inputs, dates, uregions, r):
    trg = inputs['target'][...,0]; e = trg-r
    ok = torch.isfinite(e)
    
    est = {}    
    device = inputs['yinput'].device
    month = torch.tensor([d.month for d in dates], device=device)
    for ireg,reg in enumerate(uregions+['others', 'all']):
        est[reg] = {}
        if reg in uregions:
            region = inputs['yinput'][0,:,ireg]>0.5
        elif reg == 'others':
            region = inputs['yinput'][0,:,:2].sum(-1)>0.5
        else:
            region = torch.full(inputs['yinput'].shape[1:2], True, device=device)
        for m in [1,2,3,4,5,6,12]:
            est[reg][m] = getests (inputs, r, select = month==m, region=region)[2:]
        fig, ax = plt.subplots(1,1)
        ax.set_title(reg+' BIAS')
        m = month[:,None].expand(-1,region.sum())[ok[:,region]].cpu().numpy()
        sns.boxenplot(x=m, y=e[:,region][ok[:,region]].cpu().numpy(), linewidth=1.5, ax=ax)
        fig, ax = plt.subplots(1,1)
        ax.set_title(reg+' MAE')
        sns.boxenplot(x=m, y=torch.abs(e[:,region][ok[:,region]]).cpu().numpy(), linewidth=1.5, ax=ax)
    
    fig, ax = plt.subplots(1,1)
    ax.set_title('BIAS')
    x,y = quantized (r[ok],e[ok],nrmy=False)
    sns.boxenplot(x=x,y=y, linewidth=1.5, ax=ax)
    ax.set_xlabel('forecast')
    fig, ax = plt.subplots(1,1)
    ax.set_title('MAE')
    x,y = quantized (r[ok],torch.abs(e[ok]),nrmy=False)
    sns.boxenplot(x=x,y=y, linewidth=1.5, ax=ax)
    ax.set_xlabel('forecast')
    # ok = torch.isfinite(e) & (torch.abs(e) < 15.)
    # x,y = quantized (r[ok],torch.abs(e[ok]),nrmy=False)
    # sns.boxenplot(x=x,y=y, linewidth=1.5)
    return est
