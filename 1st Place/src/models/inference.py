import gc
import numpy as np
import torch
from . import calcdevice
from models.apply import applymodels

def getests (inputs, result, factor=1., select=None, region=None):
    e = inputs['target'][...,0]-result*factor
    e1 = e[torch.isfinite(e)].detach().cpu()
    est = np.round([e1.mean().item(), np.sqrt((e1*e1).mean()).item()],4)
    if select is not None:
        e = e[select]
    if region is not None:
        e = e[:,region]
    e = e[torch.isfinite(e)].detach().cpu()
    est = np.hstack((est,np.round([e.mean().item(), np.sqrt((e*e).mean()).item()],4)))
    return est

def quantile(vals, w, q=0.5):
    s,perm = torch.sort(vals, -1)
    ws = torch.gather(w.expand_as(vals),-1,perm)
    ws = torch.cumsum(ws,-1)
    ws.div_(ws[...,-1:])
    c = (ws<q).sum(-1,keepdim=True).clamp_(ws.shape[-1]-2)
    w1 = torch.gather(ws,-1,c)
    w2 = torch.gather(ws,-1,c+1)
    return ((torch.gather(s,-1,c)*(w2-q) + torch.gather(s,-1,c+1)*(q-w1))/(w2-w1))[...,0]*w.sum(-1)

# result = (rets1*w[:,None]).sum(-1)
# result = quantile(rets1, w[:,None])
# w = rets - torch.nan_to_num (x['target'])
# L2 = 1.
# w = 1./ ((w*w).clamp(max=10.).sum(1)/okx.sum(1) + L2*L2)
# # w.mul_((w<w.max(-1,keepdim=True)[0])&(w>w.min(-1,keepdim=True)[0]))
# w.div_(w.sum(-1,keepdim=True))
# r = (rets*w[:,None]).sum(-1,keepdim=True)
# w.mul_ ((r*torch.nan_to_num (x['target'])).sum(1)/(r*r).sum(1)).mul_ (0.825)
# result = (rets1*w[:,None]).sum(-1)
# models.getests (inputs['test'], result, select=spring)
# trg = inputs['test']['target'][...,0]
# ok  = torch.isfinite(trg-result); r = result[ok]
# bfactor = ((trg[ok]*r).sum()/(r*r).sum()).item()
# models.getests (inputs['test'], result, bfactor, select=spring)

# xv = inputs['test']['xval'].clone()
# inputs['test']['xval'][bad] = np.nan
# rets11 = models.applymodels (inputs['test'], modelslist, average=False).clamp_(min=0.)
# inputs['test']['xval'] = xv

def inference (inputs, models, dates, lab='', smart=True, L2=1., factor=1., device=calcdevice,
               calcsd=False, noemb=False, nousest=True, print=print):
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        if smart is not None and 'ylo' in inputs:
            x = {key: inputs[key].detach() for key in inputs if key[0] == 'x'}
            x['yid'] = x['xid'] = torch.zeros_like(x['xlo'])
            x['target'] = x['xval'][...,:1].clone()
            trg = x['target'].clone()
            okx = torch.isfinite(trg)
            rets = applymodels (x, models, average=False, device=device)*okx            
            w = rets - torch.nan_to_num (trg) 
            w = 1./ ((w*w).sum(1)/okx.sum(1) + L2*L2)
            # w = 1./ (torch.abs(w).sum(1)/okx.sum(1) + L2)
            # kf = w.max(-1,keepdim=True)[0]/w.min(-1,keepdim=True)[0]
            # print ([kf.max(), kf.mean(), kf.min()])
            # w.mul_((w<w.max(-1,keepdim=True)[0])&(w>w.min(-1,keepdim=True)[0]))
            w.div_(w.sum(-1,keepdim=True))
            r = (rets*w[:,None]).sum(-1,keepdim=True)
            # if qc:
            #     bad = ((trg[...,0]<rets.min(-1)[0]-3.)&(trg[...,0]>3.)) | (trg[...,0]>rets.max(-1)[0]+3.)
            #     print(f"qc #{bad[torch.isfinite(trg)[...,0]].float().mean().item()*100.:.5}%")
            w.mul_ ((r*torch.nan_to_num (trg)).sum(1)/(r*r).sum(1))
            # if nousest:
            w.mul_ (0.875) #Magic constant
            # else:
            #     w.mul_ (0.925) #Magic constant
            del x, rets, okx
            gc.collect()
            torch.cuda.empty_cache()
            # if qc:
            #     xv = inputs['xval'].clone()
            #     inputs['xval'][bad] = np.nan
            w = w[:,None]
            if noemb and 'yemb' in inputs:
                ws = w.sum(-1,keepdim=True)
                w = w.expand(-1,inputs['yval'].shape[1],-1)
                noembmodels = torch.tensor([models[m].embedding<=0 for m in models], device=w.device)
                w = w*((inputs['yemb'][0,:,None]>0)|noembmodels)
                w = w/w.sum(-1,keepdim=True)*ws
            if nousest:
                print('Noemb mult 0.85')
                w = w*(1.-0.15*(inputs['yemb'][0,:,None]==0))
            result = applymodels (inputs, models, average=w, device=device, calcsd=calcsd).clamp_(min=0.)
            # if qc:
            #     inputs['xval'] = xv
        else:
            result = applymodels (inputs, models, device=device, calcsd=calcsd).clamp_(min=0.)
            result.mul_ (0.9)
        # result [result<0.01] = 0.
    if 'target' in inputs:
        spring = np.array([d.month*100+d.day for d in dates])
        spring = (spring>=215)&(spring<=701)#&np.array([d.year>=2021 for d in dates])
        spring = torch.tensor(spring, device=result.device)
        # spring = None
        exper = torch.isfinite(inputs['target'][...,0]).float().mean(0)<0.2
        ests = getests (inputs, result, select=spring)
        msg = lab+(f' L2={L2}' if smart else '')+(' sd' if calcsd else '')+('' if nousest else '|new|')+f' raw={ests}'        
        ests = getests (inputs, result, select=spring, region=exper)
        msg += f'{ests[2:]}'
        if factor != 1.:
            ests = getests (inputs, result, factor, select=spring)
            msg += f' factor={factor:.4}:{ests}'        
        trg = inputs['target'][...,0]
        ok  = torch.isfinite(trg-result); r = result[ok]
        bfactor = ((trg[ok]*r).sum()/(r*r).sum()).item()
        ests = getests (inputs, result, bfactor, select=spring)
        msg += f' bfactor={bfactor:.4}:{ests}'
        ests = getests (inputs, result, bfactor, select=spring, region=exper)
        msg += f'{ests[2:]}'
        print('Estimations: '+ msg)
    result = result.mul_(factor) #.to(maindevice)
    return result

def test(inputs, models, dates, lab='', iyear='', device=calcdevice, calcsd=False, print=print, nousest=True):
    models = {key: models[key].to(device) for key in models}
    if len(models) == 1:
        calcsd = False
    r = {mode: inference (inputs[mode], models, dates[mode], lab=lab+'_'+mode+iyear, smart=True, factor=1., device=device, calcsd=calcsd, print=print, nousest=nousest) for mode in inputs}
    if 'test' in inputs:
        inference (inputs['test'], models, dates['test'], lab=lab+'_test'+iyear, smart=None, device=device, calcsd=calcsd, print=print, nousest=nousest)
        # if len(models) > 1:
        #     inference (inputs['test'], models, dates['test'], lab=lab+'_test'+iyear, smart=None, device=device, calcsd=~calcsd, print=print, nousest=nousest)
    return r