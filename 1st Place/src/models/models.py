from . import functions
import numpy as np
from models.oanet4s import DLOA, Parameter, Model0
import torch

getstyle = lambda style: 'season' if style is None else style

class Calibration(torch.nn.Module):
    def __init__(self, **kwargin):
        torch.nn.Module.__init__(self)
        # self.p = Parameter([ 4.,  0., -4., -0.5, -4., -3.])
        self.p = Parameter([ 3.,  0., -3., -0.5, -5.])
    def forward(self,r):
        p = self.p.to(r.device)
        r1 = r/10.
        if p.numel() == 5:
            k = p[2]+r1*(p[3]+r1*p[4])        
        else:
            k = p[2]+r1*(p[3]+r1*(p[4]+r1*p[5]))
        return (r+p[0].clamp(max=-p[2])+k*torch.exp(-r1*functions.sigma_act.apply(p[1]))).clamp_(min=0.)

class Model(DLOA):
    def __init__(self, ninputs, nfunc, nlatent=3, norm=False, nmonths=1, calibr=False,
                  initgamma=1., embedding=0, nstations=1, style=None, mc=1, dropout=0., initemb=None, **kw):
        kw['neddvalgrad'] = initgamma is not None        
        self.style = getstyle(style)
        if self.style[-3:] == 'pos':
            nfunc *= 2
        # DLOA.__init__(self, 2*ninputs+embedding, nfunc, nlatent, **kw)
        # self.m0 = DLOA(0,1,0,implicit=False,edging=True,rmax=1.,usee=False,biased=False,relativer=True)
        DLOA.__init__(self, ninputs+embedding, nfunc, nlatent, **kw)
        self.calcsd = self.dist.calcsd
        self.embedding = embedding
        self.mc = mc
        self.nmonths = nmonths
        if embedding > 0:
            emb = torch.nn.Embedding(nstations*nmonths, embedding, max_norm=2.)
            if initemb is None:
                emb.weight.data.uniform_(-0.15, 0.15)
                emb.weight.data[:nmonths] = 0
            else:
                emb.weight.data[:,0] = initemb            
            if dropout > 0:
                self.emb = torch.nn.Sequential(emb, torch.nn.Dropout(dropout))
            else:
                self.emb = torch.nn.Sequential(emb)
            self.embmult = 0.2
            # self.net[1].weight.data[:,len(arglist):len(arglist)+self.embedding] = \
            #     self.embmult*self.net[1].weight.data[:,len(arglist):len(arglist)+self.embedding]
        self.calibr = None
        if calibr:
            self.calibr = Calibration()
        self.norm = norm
        if initgamma is not None:
            self.gamma = Parameter(initgamma)      
    def forward(self, xx, isval=True):
        sel = torch.isfinite(xx['xval'][...,0]).sum(1) > self.points
        select = (~sel).sum() > 0
        if sel.sum() > 0:
            x = {key: xx[key][sel] for key in xx} if select else xx
        else:
            return torch.full(xx['ylo' if 'ylo' in xx else 'xlo'][...,None].shape, np.nan, device=xx['xval'].device),None
        self.dist.calcsd = self.calcsd
        xval = x['xval'].clone()
        usey = 'ylo' in x
        xi = x['xinput'].clone()
        if usey:
            yi = x['yinput'].clone()
        # embinterp = (not self.training) and (self.embedding > 0)
        # # embinterp = self.embedding > 0
        # if embinterp:
        #     xemb = torch.cat((x['xemb'],x['yemb']),1) if usey else x['xemb']
        #     device = xemb.device
        #     isy = xemb[0] == 0
        #     embedding = self.embedding; self.embedding = 0
        #     if isy.sum() > 0:
        #         isx = ~isy
        #         if isx.sum() > 0:
        #             xe = {'x'+key: torch.cat((x['x'+key],x['y'+key]),1) if usey else x['x'+key] for key in ['lo', 'la']}
        #             xx = {key: xe[key][:,isx] for key in xe}
        #             xx.update({'y'+key[1:]: xe[key][:,isy] for key in xe})                
        #             res = torch.zeros(xe['xlo'].shape+(embedding,), device=device)
        #             xx['xval'] = self.emb(xemb[:,isx]).mul_(self.embmult)
        #             res[:,isx] = xx['xval']
        #             res[:,isy] = Model0.to(device) (xx)
        #             x['xinput'] = torch.cat([x['xinput'],res[:,:x['xinput'].shape[1]]],-1)
        #             if usey:
        #                 x['yinput'] = torch.cat([x['yinput'],res[:,x['xinput'].shape[1]:]],-1)
        #         else:
        #             x['xinput'] = torch.cat([x['xinput'],torch.zeros(x['xlo'].shape+(embedding,), device=device)],-1)
        #             if usey:
        #                 x['yinput'] = torch.cat([x['yinput'],torch.zeros(x['ylo'].shape+(embedding,), device=device)],-1)
        #     else:
        #         x['xinput'] = torch.cat([x['xinput'],self.emb(x['xemb'])],-1)
        #         if usey:
        #             x['yinput'] = torch.cat([x['yinput'],self.emb(x['yemb'])],-1)
        sd = None
        if isval:
            xv = x['xval']/10. if self.biased else x['xval']
            bad = torch.isnan(xv)
            if hasattr(self, 'gamma'):
                gamma = torch.sigmoid(self.gamma).to(xv.device)
                ok = (~bad)&(xv>0.)
                xv[ok] = torch.pow(xv[ok], gamma)       
            if self.norm:
                xv  = torch.nan_to_num(xv,0.)
                nrm = (xv.sum(1,keepdim=True)/(xv>0.).float().sum(1,keepdim=True).clamp_(min=1.)).clamp_(min=0.1)
                xv = xv/nrm
            if (self.mc > 1) and self.training and (not self.net[0].frozen):
            # if self.mc > 1:
                res = []
                if self.training:
                    rnd = torch.randn_like(xv)
                for f in [-0.15,0.15,0.0][:self.mc]:
                    # f1 = f*(1.-self.net[0].factor)
                    x['xval'] = xv*((f*rnd + 1.).clamp_(min=0.,max=2.) if self.training else (f + 1.))
                    x['xval'][bad] = np.nan
                    res.append(DLOA.forward(self,x)[...,None])
                res = torch.cat(res,-1).mean(-1)
            else:
                xv[bad] = np.nan
                x['xval'] = xv
                if self.style[-3:] == 'pos':
                    x['xval'] = torch.cat((x['xval'],torch.tanh(xv)),-1)
                res = DLOA.forward(self,x)
            if self.norm:
                if self.style[-3:] == 'pos':
                    res = res[...,:xv.shape[-1]].clamp(min=0.)*(2.*res[...,xv.shape[-1]:]-1.).clamp(min=0.,max=1.)*nrm
                else:
                    res = res*nrm
        else:
            res = DLOA.forward(self,x)
        if isval:
            if hasattr(self, 'gamma'):
                ok = torch.isfinite(res)&(res>0.)
                res[ok] = torch.pow(res[ok], 1./gamma)
            if self.biased:
                res = (res*10.).clamp(min=0.)
            if self.calibr is not None:
                res = self.calibr(res)
            x['xval'] = xval
        # if embinterp:
        #     self.embedding = embedding
        x['xinput'] = xi
        if usey:
            x['yinput'] = yi
        if 'sd' in x:
            for key in ['sd','esd','ysigma','yval']:
                if key in x:
                    x[key] = x[key][...,:1]*nrm[...,:1] if self.norm else x[key][...,:1]
                    if hasattr(self, 'gamma'):
                        x[key] = torch.pow(x[key], 1./gamma)
            sd = x['sd']
        if select:
            res1 = torch.full(xx['ylo' if 'ylo' in xx else 'xlo'].shape+res.shape[-1:], np.nan, device=xx['xval'].device)
            res1[sel] = res            
            if sd is not None:
                sd1 = torch.full(xx['ylo' if 'ylo' in xx else 'xlo'].shape+(1,), np.nan, device=xx['xval'].device)
                sd1[sel] = sd
                return res1,sd1
            return res1,None
        return res,sd
    def freeze(self, frozen):
        self.dist.kf.frozen = frozen
        if self.usee:
            self.enet.kf.frozen = frozen
        self.net[0].frozen = frozen>0
    def prints(self):
        return DLOA.prints(self) + (f' gamma={torch.sigmoid(self.gamma).item():.4}' if hasattr(self, 'gamma') else '')
    def kfparams(self):
        return [p for p in self.dist.parameters() if p.requires_grad]+\
              ([p for p in self.enet.parameters() if p.requires_grad] if self.usee else [])+\
              ([self.gamma] if hasattr(self, 'gamma') else [])+\
              ([self.calibr.p] if self.calibr is not None else [])
    def netparams(self):
        return [p for p in self.net.parameters() if p.requires_grad]+\
              ([p for p in self.emb.parameters() if p.requires_grad] if self.embedding > 0 else [])
    def importance(self, arglist):
        w = self.net[1].weight; w = torch.sqrt((w*w).sum(0))
        roundw = lambda x: np.round(x.item(),3) if x.numel() == 1 else list(np.round(x.detach().cpu().numpy(),3))
        imp = {key: roundw(w[i]) for i,key in enumerate(arglist)}
        if self.embedding > 0:
            imp['emb'] = roundw(w[len(arglist):len(arglist)+self.embedding])
        w = w[len(arglist)+self.embedding:]
        if self.densed:
            nkf = self.dist.kf.rm.shape[0]
            imp.update({'fg': roundw(w[:-nkf]), 'dense': roundw(w[-nkf:])})
        else:
            imp['fg'] = roundw(w)
        return imp
    def nonzero(self):
        nz = {p.weight: torch.ones_like(p.weight,dtype=torch.bool) for p in self.net[1:-1:2]}
        if self.embedding > 0 and self.nmonths > 1:
            nz.update ({p: torch.ones_like(p,dtype=torch.bool) for p in self.emb.parameters() if p.requires_grad})
        return nz
              
class Model3(torch.nn.Module):
    def __init__(self, weights, *argin, layers=2, style=None, mc=1, commonnet=True, **kwargin):
        torch.nn.Module.__init__(self)
        self.style = getstyle(style)
        # mc=1
        nets = [Model(*argin, style=style, mc=mc, **kwargin)]
        nnets = {'season': 1, 'season_pos': 1}
        if self.style == 'stack':
            kwargin.update({'usee': False, 'rmax': kwargin['rmax2']}) #, 'implicit': True
            nets += [Model(argin[0]+argin[1], *argin[1:], style=style, mc=mc, **kwargin) for k in range(layers-1)]
        else:
            if self.style == 'pos':
                # kwargin.update({'initgamma': None, 'sigmed': False})
                kwargin.update({'initgamma': None})
            nets += [Model(*argin, style=style, mc=mc, **kwargin) for k in range(nnets[self.style] if self.style in nnets else layers-1)]
        self.nets = torch.nn.ModuleList(nets)
        self.calcsd = self.nets[0].dist.calcsd
        self.commonnet = self.style not in ['stack'] and commonnet
        # self.commonnet = False
        if weights is not None:
            weights = {key: weights[key].clone() for key in weights}
            if self.style == 'season' and 'gamma' in weights:
                weights['gamma'].add_(0.2)
            self.nets[0].load_state_dict(weights)
            if self.style not in ['stack']:
                if self.style[:6] == 'season' and 'gamma' in weights:
                    weights['gamma'].sub_(0.4)
                if self.nets[1].sigmed:
                    for net in self.nets[1:]:
                        net.load_state_dict(weights)
        if self.style not in ['stack'] and self.commonnet:
            for net in self.nets[1:]:
                net.net = self.nets[0].net
                if self.nets[0].embedding > 0:
                    net.emb = self.nets[0].emb
        self.points = self.nets[0].points
    def nonzero(self, nzero=None):
        getnz = lambda net: net.nonzero().keys()
        if nzero is None:
            nzero = self.nets[0].nonzero()
        nz = {p: nzero[p0] for (p,p0) in zip(getnz(self.nets[0]),nzero.keys())}
        for net in self.nets[1:]:
            if self.style == 'stack':
                nz.update ({p: torch.ones(p.shape,dtype=torch.bool,device=p.device) for (p,p0) in zip(getnz(net),nzero.keys())})
            else:
                nz.update ({p: nzero[p0] for (p,p0) in zip(getnz(net),nzero.keys())})
        return nz        
    def forward(self, xx):
        for net in self.nets:
            net.points = self.points
            net.dist.calcsd = self.calcsd
        sel = torch.isfinite(xx['xval'][...,0]).sum(1) > self.points
        select = (~sel).sum() > 0
        if sel.sum() > 0:
            x = {key: xx[key][sel] for key in xx} if select else xx
        else:
            return torch.full(xx['ylo' if 'ylo' in xx else 'xlo'][...,None].shape, np.nan, device=xx['xval'].device),None
        sd = None
        if self.style == 'stack':
            res,sd = self.nets[0](x)
            xin = x['xinput'].clone()
            if 'ylo' in x:
                yin = x['yinput'].clone()
                x['xinput'] = torch.cat([x['xinput'], x['xval']], -1)
            yinp = 'yinput' if 'ylo' in x else 'xinput'
            x[yinp] = torch.cat([x[yinp], res], -1)
            res,sd = self.nets[1](x)
            for net in self.nets[2:]:
                x[yinp][...,-res.shape[-1]] = res
                res,sd = net(x)
            x['xinput'] = xin
            if 'ylo' in x:
                x['yinput'] = yin
            return res,sd
        elif self.style[:6] == 'season':
            rets = []
            sd = []
            for i in range(len(self.nets)):
                # x['xinputs2'] = i
                # x['yinputs2'] = i
                rets.append(self.nets[i](x)[0][...,None])
                if 'sd' in x:
                    sd.append(x['sd'])
            rets = torch.cat(rets,-1)
            d = rets.mean(-1)
            d = torch.sigmoid(d[...,:1]-d[...,-1:])
            res = (rets[...,:1,0]-rets[...,:1,1])*d+rets[...,:1,1]
            if len (sd) > 0:
                # printshape(sd+[d])
                sd = (sd[0]-sd[1])*d+sd[1]
            else:
                sd = None
        else:
            res = torch.cat([net(x)[...,None] for net in self.nets],-1).mean(-1)
        if select:
            res1 = torch.full(xx['ylo' if 'ylo' in xx else 'xlo'][...,None].shape, np.nan, device=xx['xval'].device)
            res1[sel] = res            
            if sd is not None:
                sd1 = torch.full(xx['ylo' if 'ylo' in xx else 'xlo'][...,None].shape, np.nan, device=xx['xval'].device)
                sd1[sel] = sd
                return res1,sd1
            return res1,None
        return res,sd
    def freeze(self, frozen):
        for net in self.nets:
            net.freeze(frozen)
    def prints(self):
        return ' '.join([net.prints() for net in self.nets])
    def kfparams(self):
        r = []
        for par in [net.kfparams() for net in self.nets]:
            r += par
        return r
    def netparams(self):                        
        p = [net.netparams() for net in self.nets]
        r = p[0]
        if not self.commonnet:
            for par in p[1:]:
                r += par[2:] if self.style[:6] == 'season' else par
        return r
    def loss(self, *argin, **kwargin):
        return self.nets[0].loss(*argin, **kwargin)
    def importance(self, arglist):
        if self.commonnet:
            return self.nets[0].importance(arglist)
        imp = [net.importance(arglist) for net in self.nets]
        return {key: [i[key] for i in imp] for key in imp[0]}
    