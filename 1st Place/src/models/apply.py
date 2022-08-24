import numpy as np
import torch
from models import calcdevice

valid_bs = 8

def apply(m, inputs, sel, device=calcdevice, autoloss=False, lab='', lossfun=None, selecty=None, calcsd=False):     
    x = {key: inputs[key][sel].detach() for key in inputs if key[0] != 'i'}
    x['xid'] = torch.zeros_like(x['xlo'])
    x['yid'] = torch.ones_like (x['ylo']) if 'ylo' in x else x['xid']
    okx = torch.isfinite(x['xval'][..., 0]).any(0)
    if m.training:
        oky = torch.isfinite(x['yval'][..., 0]).any(0)        
        x = {key: x[key][:,okx if key[0]=='x' else oky].to(device).detach() for key in x}
        if autoloss:
            if selecty is None:
                selecty = inputs['isstation'].to(device)
            xx = {'x'+key: torch.cat((x['y'+key][:,selecty[oky]],x['x'+key]),1).detach() for key in ['val','lo','la','input','id','emb'] if 'x'+key in x}
            xx['yid'] = xx['xid']
            xx['target'] = xx['xval'][...,:1].clone()
            res = m (xx)[0][..., :1]
            sely = ~selecty[oky.to(device)]; calcyx = (sely.sum()>0).item()
            if calcyx:
                yx = {key: xx[key] for key in ['xval','xlo','xla','xinput','xid','xemb'] if key in xx}
                yx.update({key: x[key][:,sely].detach() for key in x if (key not in yx) and (key[0] != 'x')})
                eres = m (yx)[0][..., :1]
            if lossfun is not None:
                loss,cnt = m.loss(xx, res, lab=lab, lossfun=lossfun)
                if calcyx:
                    eloss,ecnt = m.loss(yx, eres, lab=lab, lossfun=lossfun)
                    return res, loss+eloss, cnt+ecnt
                return res, loss, cnt
    elif 'ylo' in x:        
        if selecty is not None:
            oky = torch.isfinite(x['yval'][..., 0]).any(0)
            x = {key: x[key][:,okx if key[0]=='x' else oky&selecty.to(oky.device)].to(device).detach() for key in x}
        else:
            x = {key: (x[key][:,okx] if key[0]=='x' else x[key]).to(device).detach() for key in x}
    else:
        x = {key: x[key].to(device).detach() for key in x}
    res,sd = m (x)
    res = res[..., :1]
    if sd is not None:
        if calcsd:
            return res,sd
        else:
            x['sd'] = sd[...,:1]
    if lossfun is not None:
        loss,cnt = m.loss(x, res, lab=lab, lossfun=lossfun)
        return res, loss, cnt
    return res

def applymodels (inputs, models, average=True, device=calcdevice, calcsd=False):
    sh = inputs['ylo' if 'ylo' in inputs else 'xlo'].shape
    arxdevice = inputs['xlo'].device
    batches = int(np.ceil(sh[0]/valid_bs))
    for iyear in models:
        models[iyear].calcsd = 2 if calcsd else 0
    with torch.no_grad():
        if average is True:
            average = torch.full([len(models)], 1./len(models), device=arxdevice)
        if isinstance(average,torch.Tensor):
            result = torch.zeros(sh, device=arxdevice)
            # if calcsd:
            #     std = torch.zeros(sh, device=arxdevice)
            for i in range (batches):
                sel = torch.arange(i*valid_bs, min((i+1)*valid_bs,sh[0]), device=arxdevice)
                av = average[sel] if average.ndim>1 else average
                # torch.cuda.empty_cache()
                res = 0.
                if calcsd:
                    sd = 0.
                    for i,iyear in enumerate(models.keys()):
                        r,s = apply(models[iyear], inputs, sel, device=device, calcsd=calcsd)
                        s = s[...,0].to(arxdevice)
                        res = r[...,0].to(arxdevice)/s*av[...,i] + res
                        sd = 1./s*av[...,i] + sd
                    res = res*av.sum(-1)/sd
                    # std[sel] = sd
                else:
                    for i,iyear in enumerate(models.keys()):
                        res = apply(models[iyear], inputs, sel, device=device, calcsd=calcsd)[...,0].to(arxdevice)*av[...,i] + res
                result[sel] = res
        else:
            result = torch.zeros(sh+(len(models),), device=arxdevice)
            if calcsd:
                std = torch.zeros(sh+(len(models),), device=arxdevice)
                for i in range (batches):
                    sel = torch.arange(i*valid_bs, min((i+1)*valid_bs,sh[0]), device=arxdevice)
                    for i,iyear in enumerate(models):
                        r,s = apply(models[iyear], inputs, sel, device=device, calcsd=calcsd)
                        result[sel,...,i] = r[...,0].to(arxdevice)
                        std[sel,...,i] = s[...,0].to(arxdevice)
            else:
                for i in range (batches):
                    sel = torch.arange(i*valid_bs, min((i+1)*valid_bs,sh[0]), device=arxdevice)
                    result[sel] = torch.cat([apply(models[iyear], inputs, sel, device=device, calcsd=calcsd).detach() for iyear in models], -1).to(arxdevice)
    if calcsd and average is False:
        return result,std
    else:
        return result