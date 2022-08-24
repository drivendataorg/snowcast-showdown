from . import addons
from . import functions
import numpy as np	
import torch
from torch import nn
import matplotlib.pyplot as plt
try:
    from torch_scatter import scatter
except:
    scatter = None

Parameter = lambda x: nn.Parameter(x.float().requires_grad_(True), requires_grad=True) if isinstance(x,torch.Tensor) \
                      else nn.Parameter(torch.tensor(x, dtype=torch.float32, requires_grad=True))
Variable  = lambda x: torch.autograd.Variable(x if isinstance(x,torch.Tensor) else torch.tensor(x), requires_grad=False)

def MLP(ninputs, nf, noutputs, bias=True, sgd2=False):
    layers = [addons.Normalize([ninputs], dims=[0])]
    Linear = addons.Linear2 if sgd2 else nn.Linear
    for nin,nout in zip([ninputs]+nf[:-1],nf):
        layers += [Linear (nin,nout), nn.ReLU(inplace=True)]
        # layers += [Linear (nin,nout), nn.Tanh()]
    layers.append (Linear (nf[-1] if len(nf)>0 else ninputs, noutputs, bias=bias))
    net = nn.Sequential (*layers)
    net.apply(lambda m: addons.weight_init(m,gain=0.9))
    net[-1].weight.data.mul_(0.5/np.sqrt(noutputs))
    return net

# bigdist = 256.*256.
bigdist = 1024.
def distancesq(lo1, la1, lo2, la2):
    with torch.no_grad():
        dlo = torch.remainder(torch.abs(lo1-lo2)+180.,360.)-180.
        dlo.mul_(torch.cos((la1+la2)*0.00872665)); dlo.mul_(dlo)
        dla = la1-la2; dla.mul_(dla).add_(dlo)
        return torch.nan_to_num_(dla,bigdist)

class nearestf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lo1, la1, lo2, la2, points, map1=None, map2=None, exclude=0, bad=None):
        with torch.no_grad():
            sh = lo2.shape
            s1 = (sh[0]+4)*sh[1]*lo1.shape[1]
            if map1 is not None and map2 is not None:
                s1 *= 1+map1.shape[-1]
            splits = int(np.ceil(s1/5.0e8))
            sz     = int(np.ceil(sh[1]/splits))
            inds = []; diss=[]
            for i in range(splits):
                i0 = i*sz
                i1 = (i+1)*sz if i < splits-1 else lo2.shape[1]
                dis = distancesq(lo2.data[0,i0:i1,None], la2.data[0,i0:i1,None], lo1.data[0,None], la1.data[0,None])
                dis = dis[None]+(bad.to(dis.dtype)*bigdist)[:,None].to(dis.dtype) if bad is not None else dis[None,:,:]
                if map1 is not None and map2 is not None:
                    d = map1.data[:,None]-map2.data[:,i0:i1,None]
                    dis = torch.einsum ('k...j,k...j->k...', d, d).add_(dis)
                dis,ind = torch.topk(dis, points+exclude, dim=-1, largest=False)
                if exclude>0:
                    ind = ind[..., exclude:];  dis = dis[..., exclude:]
                if ind.shape[0] != sh[0]:
                    ind = ind.expand(sh[0],-1,-1)
                    dis = dis.expand(sh[0],-1,-1)
                diss.append(dis);  inds.append(ind)
            if splits>1:
                ind = torch.cat(inds,dim=1)
                dis = torch.cat(diss,dim=1)
        ctx.save_for_backward(map1,map2,ind)
        return ind.detach(),dis.detach()
    @staticmethod
    def backward(ctx, gind, gdis):
        dm1=None; dm2=None
        if ctx.needs_input_grad[5]:
            map1,map2,ind = ctx.saved_tensors
            ind1 = ind.reshape(*((ind.shape[0],-1)+(1,)*(map1.ndim-2)))
            # d = map2.unsqueeze(-2)
            d = map2.unsqueeze(ind.ndim-1) - torch.gather(map1, 1, ind1.expand(-1,-1,*map1.shape[2:])).reshape(*ind.shape,*map1.shape[2:])
            d.mul_(gdis.unsqueeze(-1)*2.)
            if ctx.needs_input_grad[6]:
                dm2 = d.sum(ind.ndim-1)
            d = d.reshape(d.shape[0], -1, *d.shape[2-map1.ndim:])
            if scatter:
                dm1 = scatter (d, ind1, dim=1, dim_size=map1.shape[1], reduce="sum")
            else:
                dm1 = torch.zeros_like(map1)
                dm1.scatter_add_(1, ind1.expand(-1,-1,*d.shape[2-map1.ndim:]), d)
            dm1 = -dm1
        return None,None,None,None,None,dm1,dm2,None,None
        
class distsq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, dim=-1):        
        ctx.save_for_backward(x, y); d = x-y
        ctx.dim = dim
        return torch.einsum ('k...j,k...j->k...' if dim == -1 else '...jk,...jk->...k', d, d).detach()
    @staticmethod
    def backward(ctx, gdis):
        dx=None; dy=None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            x,y = ctx.saved_tensors
            d = (x - y)*gdis.unsqueeze(ctx.dim)
            if ctx.needs_input_grad[0]:
                xdim = [i for i in range(d.ndim) if d.shape[i]>x.shape[i]]
                dx = d.sum(xdim, keepdim=True).mul_(2.) if len (xdim) > 0 else d*2.
            if ctx.needs_input_grad[1]:
                ydim = [i for i in range(d.ndim) if d.shape[i]>y.shape[i]]
                dy = (d.sum(ydim, keepdim=True) if len (ydim) > 0 else d).mul_(-2.)
        return dx,dy,None

class MyKF(nn.Module):
    def __init__(self, nfunc, rmax=1., *argin, **kwargin):
        nn.Module.__init__(self)
        self.rm = Parameter(torch.zeros(nfunc,1))
        self.rmax2 = rmax**2 
        # self.relativer = relativer
        self.frozen = 0
    def prints(self):
        r = torch.sqrt(self.getrm()).data.cpu().numpy()
        if r.size > 1:
            return f' Rg: [{111./r[0,0]:.4} {111./r[-1,0]:.4}]'
        return f' Rg: {111./r[0,0]:.4}'
    def getrm(self):
        return functions.sigma_act.apply(self.rm)/self.rmax2
# calc KF (sqrt(d))
    def forward(self, d, exp):
        C = functions.xp1expm.apply(d.reshape(1,-1)*self.getrm().to(d.device), exp)
        return C.reshape(*C.shape[:-1],*d.shape)

class DistOA(nn.Module):
    def __init__(self, nfunc, edging=False, implicit=True, eps=-1.5, neddvalgrad=False, calcsd=0, *argin, **kwargin):
        nn.Module.__init__(self)
        self.implicit = implicit
        self.edging = edging
        self.logsigmoid = torch.nn.LogSigmoid()
        self.kf = MyKF(nfunc, *argin, **kwargin)
        self.softdot = functions.softdot1.apply if neddvalgrad else functions.softdot.apply
        self.neddvalgrad = neddvalgrad
        self.calcsd = calcsd
        self.lusolve = torch.linalg.solve if hasattr(torch,'linalg') and hasattr(torch.linalg,'solve') else lambda C,b: torch.solve(b,C)[0]
        self.eps = Parameter(eps*torch.ones(nfunc))
    def prints(self):
        self.kf.eval()
        eps = self.geteps(torch.device('cpu'))[...,0,0,0]
        if not self.implicit:
            eps = torch.exp(eps)
        msg = self.kf.prints()
        if self.implicit or not self.edging:
            msg += ' eps:' + np.array2string(eps.detach().cpu().numpy())
        self.kf.train(self.training)
        return msg.replace('\n','')
    def func(self, d, eval=True, sq=False):
        if eval:
            self.kf.eval()
        eps = self.geteps(d.device).squeeze(-1).squeeze(-1)
        if not self.implicit:
            eps = torch.exp(eps)
        if eps.ndim == 2:
            eps = eps[:,None]*torch.eye(eps.shape[0],device=eps.device) 
        f = self.kf (d if sq else d*d, True)
        f = f[None]*torch.eye(eps.shape[0],device=f.device)[..., None]
        if eval:
            self.kf.train(self.training)
        de = torch.diag(f[:,:,0] + eps[:,:,0])
        return f/torch.sqrt(de[None,:,None]*de[:,None,None])
    def graph(self, plot=True, rmax=None):
        if rmax is None:
            rmax = 3./np.sqrt(self.kf.getrm().min().item())
        t = torch.arange(0.,rmax,rmax/99.9,device=self.eps.device)
        f = self.func(t)
        if f.ndim > 2:
            diag = torch.eye(f.shape[0], device=f.device, dtype=torch.bool)
            f = torch.cat([f[diag], f[~diag]],0)
        f = f.cpu().detach().numpy().T
        r = 111.*t.cpu().detach().numpy()
        if plot:
            plt.plot (r, f)
        return r[:,None],f
    def solve(self, b, C):        
        if self.edging:
            n1 = 1./C.shape[-1]
            d = self.lusolve(torch.cat((torch.cat((C,torch.full(C.shape[:-1]+(1,),-n1,dtype=C.dtype,device=C.device)),axis=-1),
                            torch.cat((torch.full(C.shape[:-2]+(1,C.shape[-1]),n1,dtype=C.dtype,device=C.device),
                                       torch.zeros(C.shape[:-2]+(1,1),dtype=C.dtype,device=C.device)),axis=-1)),axis=-2),
                            torch.cat((b, torch.full(b.shape[:-2]+(1,b.shape[-1]),n1,dtype=b.dtype,device=b.device)),axis=-2))[...,:-1,:]
        else:
            d = self.lusolve(C,b)            
        wg = (torch.abs(d).sum(-2,keepdim=True)-1.).clamp_(min=0.).mul(2.).clamp_(max=1.)
        d1 = b/C.sum(-1,keepdim=True)
        return d*(1.-wg) + d1*(wg/d1.sum(-2,keepdim=True).clamp_(1e-6 if self.edging else 1.))
    def geteps(self, device):
        return (torch.sigmoid(self.eps) if self.implicit else self.logsigmoid(self.eps)).to(device)[...,None,None,None]
    def forward(self, C, val, bad=None):        
        bsh = C.shape[:-2]+(C.shape[-1]-C.shape[-2],)
        kfdims = 2
        resh = C.ndim > kfdims+1
        if resh:
            C = C.reshape(-1, *C.shape[-kfdims:])
            if val.ndim>3:
                val = val.reshape(-1, *val.shape[-2:])
                if bad is not None:
                    bad = bad.reshape(val.shape[:-1])        
        if bad is None:
            bad = torch.isnan(val).any(-1)
        with torch.no_grad():
            eye = torch.eye(C.shape[1],dtype=val.dtype,device=C.device)
            if not self.implicit:
                eye.sub_(1.)
        # self.returnsd = (self.calcsd==1 and not (self.training and (self.kf.frozen>=2))) or (self.calcsd==2)
        self.returnsd = (self.calcsd==1 and self.training and (self.kf.frozen==0)) or (self.calcsd==2)
        if self.implicit or not self.edging:
            eps = self.geteps(C.device)*eye
        val0 = (val.clone() if self.neddvalgrad else val.detach()).nan_to_num_(0.)
        C = self.kf(C, self.implicit)
        
        b = C[..., C.shape[-2]:]; C = C[..., :C.shape[-2]]
        if self.implicit:
            C.add_(eps)            
            if bad.any():
                if self.training:
                    b = b*~bad.unsqueeze(-1)
                    C = C*~torch.logical_xor(bad.unsqueeze(-1),bad.unsqueeze(-2))
                else:
                    b.mul_(~bad.unsqueeze(-1))
                    C.mul_(~torch.logical_xor(bad.unsqueeze(-1),bad.unsqueeze(-2)))
            r = self.solve(b, C)
            if self.returnsd:
                d = 1.-torch.einsum ('k...ip,k...ip->...pk', r, b)
            r = torch.einsum ('k...jp,...jk->...pk', r, val0)
        else:
            if self.training:
                if not self.edging:
                    C = C - eps
                if bad.any():
                    b = b.add(bad.unsqueeze(-1), alpha=-bigdist)
                    C = C.add(torch.logical_xor(bad.unsqueeze(-1),bad.unsqueeze(-2)), alpha=-bigdist)
            else:
                if not self.edging:
                    C.sub_(eps)
                if bad.any():
                    b.add_(bad.unsqueeze(-1), alpha=-bigdist)
                    C.add_(torch.logical_xor(bad.unsqueeze(-1),bad.unsqueeze(-2)), alpha=-bigdist)
            r = b-torch.logsumexp(C,dim=-1,keepdim=True)
            if self.returnsd:
                r = nn.functional.log_softmax(r,dim=-2) if self.edging else r-torch.logsumexp(r,dim=-2,keepdim=True).clamp(min=0.)
                d = 1.-torch.exp(r+b).sum([-2,-1]).T
            r = self.softdot(r.permute(1,2,3,0), val0.unsqueeze(-2), 1, None if self.edging else 1.)
        if resh:
            r = r.reshape(*bsh[:2],-1)
        if self.returnsd:
            if resh:
                d = d.reshape(*bsh[:2],-1)
            return r,torch.sqrt(d.clamp(min=0.,max=1.))
        return r
            
def gather (x, lab):
    return (torch.gather(x[lab], 1, x['ind1']).reshape(x['ind'].shape) if x[lab].ndim<3 else
            torch.gather(x[lab], 1, x['ind1'][(...,) + (None,)*(x[lab].ndim-2)].expand(-1,-1,*x[lab].shape[2:])).reshape(*x['ind'].shape, *x[lab].shape[2:]))
def gathera (x, lab, index='ind'):
    return x[lab].reshape(-1,*x[lab].shape[2:])[x[index]]
 
def printshape(a, prefix=''):
    print (prefix+str({key: (a[key].dtype,a[key].shape,torch.isfinite(a[key]).sum().item() if isinstance(a[key],torch.Tensor) else np.isfinite(a[key]).sum()) for key in a} \
           if isinstance(a,dict) else [(x.dtype,x.shape,torch.isfinite(x).sum().item() if isinstance(x,torch.Tensor) else np.isfinite(x).sum()) for x in a]))
    
class DLOA(nn.Module):
    def __init__(self, ninputs, nfunc, nlatent=3, individual=False, usee=True, sigmed=False, reselect=True,
                 nf=[32], netinit=None, points=16, densed=False, neddvalgrad=False,
                 calcsd=0, rmax=1., rmaxe=None, sgd2=True, biased=False, relativer=False, gradloss=0.,
                 *argin, **kwargin):
        nn.Module.__init__(self)
        self.sigmed = sigmed
        self.biased = biased
        self.individual = individual
        nkf = nfunc if self.individual else 1        
        self.neddvalgrad = neddvalgrad
        self.relativer = relativer
        self.usee = usee
        self.densed = usee and densed
        self.reselect = reselect or not self.usee
        self.embedding = 0
        self.gradloss = gradloss
        self.dist = DistOA(nkf, neddvalgrad=self.biased or self.sigmed or neddvalgrad, calcsd=calcsd, 
                           rmax=rmax,#+np.sqrt(nlatent*0.5),
                           *argin, **kwargin)
        if usee:
            kwargin.update({'implicit': False, 'edging': True, 'eps': -4. if densed else -2.})
            self.enet = DistOA(nkf, neddvalgrad=False, calcsd=2 if self.densed else calcsd, 
                               rmax=rmax if rmaxe is None else rmaxe, *argin, **kwargin)
        self.nfunc = nfunc
        self.nlatent = nlatent
        self.outs = {'map': nlatent}
        if self.sigmed:
            self.outs['sigma'] = nfunc        
        if self.biased:
            self.outs['bias'] = nfunc        
        if netinit is None:
            netinit = lambda nin,nout,bias: MLP (nin, nf, nout, bias, sgd2=sgd2)
        netargs = ninputs+(nfunc+(nkf if self.densed else 0) if self.usee else 0)
        netouts = sum([self.outs[key] for key in self.outs])
        if netargs>0 and netouts>0:
            self.net = netinit (netargs, netouts, bias=self.biased or self.sigmed)
            if self.sigmed:
                self.net[-1].bias.data = torch.cat([torch.ones(self.outs[key])*(0. if key=='eps' else ( 2.)) for key in self.outs]).detach()
        else:
            self.net = None
        self.points = points
        self.history = {}
    def prints(self):
        return (self.dist.prints() + (self.enet.prints() if self.usee else '')).replace('\n','')
    def return_KF(self):
        return False
    def graph(self, *argin, **kwargin):
        return self.dist.graph(*argin, **kwargin)
    def mappingx (self, x, lab):
        if self.net is not None:
            y = [x[lab+'input']]
            if self.embedding > 0:
                x[lab+'emba'] = self.emb(x[lab+'emb' if lab+'emb' in x else 'xemb']).mul_(self.embmult)
                y.append(x[lab+'emba'])
            if self.usee:
                y.append(x[lab+'val'])
            if self.densed:
                y.append(torch.zeros(x['xval'].shape[:-1]+self.enet.eps.shape[:1],device=x['xval'].device) if lab=='x' else x['esd'])
            y = torch.cat(y,-1) if len(y)>1 else x[lab+'input']
            ok = torch.isfinite(y).all(-1)
            if ok.any():
                ymap = self.net (y[ok])
                i0 = 0
                for key in self.outs:
                    i1 = i0+self.outs[key]
                    s = ymap[..., i0:i1]; i0 = i1          
                    if key == 'sigma':
                        x[lab+key] = torch.ones (y.shape[:-1]+s.shape[-1:], dtype=ymap.dtype, device=ymap.device)                        
                        s = functions.sigma_act.apply(s)
                    else:
                        x[lab+key] = torch.zeros (y.shape[:-1]+s.shape[-1:], dtype=ymap.dtype, device=ymap.device)
                    x[lab+key][ok] = s
            if lab == 'x':
                x[lab+'ok'] = ok
        else:
            x[lab+'map'] = None
            if lab == 'x':
                x[lab+'ok'] = torch.isfinite(x[lab+'val']).all(-1)
    def distmatr(self, x, bad, ismap, usey):
        args = [x['xlo'], x['xla'], x['ylo' if usey else 'xlo'], x['yla' if usey else 'xla'], self.points]
        args += [x['xmap'], x['ymap']] if ismap else [None, None]
        x['ind'],b = nearestf.apply (*args, 1 if self.training or not usey else 0, bad)
        with torch.no_grad():
            x['ind1'] = x['ind'].reshape(x['ind'].shape[0],-1)
            ylo = gather(x, 'xlo'); yla = gather(x, 'xla')
            C = distancesq(ylo.unsqueeze(-2), yla.unsqueeze(-2), ylo.unsqueeze(-1), yla.unsqueeze(-1))
        return C,b

    def forward (self, x):        
        self.dist.kf.train(self.training)
        usey = 'ylo' in x
        bad = torch.isnan(x['xval']).all(-1)
        self.mappingx(x, 'x')
        w = nn.functional.softmax(torch.arange(0,-self.points//2,-1,dtype=torch.float32,device=x['xval'].device),dim=-1)
        if self.usee:
            C,b = self.distmatr (x, bad, False, usey)
            xg = gather(x, 'xval')
            Cb = torch.cat((C,b.unsqueeze(-1)),-1)
            x['yval'] = self.enet (Cb/(b[...,:w.shape[0]]*w).sum(-1,keepdim=True)[...,None] if self.relativer else Cb, xg)
            if self.enet.returnsd:
                x['yval'],x['esd'] = x['yval']
        if usey or self.usee:
            if not usey:
                x['yinput'] = x['xinput']
            self.mappingx(x, 'y')
        else:
            for key in self.outs:
                x['y'+key] = x['x'+key]
        
        reselect = self.reselect # and not self.training
        if reselect:      
            bad = (~x['xok']) if self.usee else (~x['xok']) | torch.isnan(x['xval']).any(-1)
            C,b = self.distmatr (x, bad, True, usey)
        if x['xmap'] is not None:
            xmap = gather (x, 'xmap')
            if not reselect and b is not None:
                b = distsq.apply(x['ymap'].unsqueeze(-2), xmap).add_(b)
            C.add_(distsq.apply(xmap.unsqueeze(-3), xmap.unsqueeze(-2)))

        x['xn'] = torch.nan_to_num(x['xval'], 0.)
        if self.training:
            if self.biased:
                x['xn'] = x['xn']-x['xbias']
            if self.sigmed:
                x['xn'] = x['xn']/x['xsigma']
        else:
            if self.biased:
                x['xn'].sub_(x['xbias'])
            if self.sigmed:
                x['xn'].div_(x['xsigma'])
        x['xn'][bad] = np.nan
        xg = gather (x, 'xn')
        Cb = torch.cat((C,b.unsqueeze(-1)),-1)
        yn = self.dist (Cb/(b[...,:w.shape[0]]*w).sum(-1,keepdim=True)[...,None] if self.relativer else Cb, xg, None)
        if self.dist.returnsd:
            yn,x['sd'] = yn        
        if self.training:
            if self.sigmed:
                yn = yn*x['ysigma']
                if self.dist.returnsd:
                    x['sd'] = x['sd']*x['ysigma']
        else:
            if self.sigmed:
                yn.mul_(x['ysigma'])
                if self.dist.returnsd:
                    # x['sd'].mul_(x['ysigma'])
                    x['sd'] = x['sd']*x['ysigma']
        if self.biased:
            yn.add_(x['ybias'])
        return yn
    def loss(self, x, result, lab=None, lossfun='mse'):
        dims = list(range(result.ndim-1))
        loss = torch.zeros_like(result)
        if lab is None:
            lab = 'train' if self.training else 'test'
        if lab+'_'+lossfun not in self.history:
            self.history[lab+'_'+lossfun] = []
        ok = torch.isfinite(x['target']) & torch.isfinite(result)
        cnt = ok.sum(dims).float()
        if lossfun=='hyber':
            loss[ok] = nn.functional.smooth_l1_loss (x['target'][ok], result[ok], reduction='none')
        elif lossfun=='mse':
            loss[ok] = nn.functional.mse_loss (x['target'][ok], result[ok], reduction='none')
        elif lossfun=='smape':
            loss[ok] = 2.*nn.functional.mse_loss (x['target'][ok], result[ok], reduction='none')/(x['target'][ok] + result[ok]).clamp_(min=2.)
        self.history[lab+'_'+lossfun].append ((loss.sum()/cnt.sum()).item())
        # printshape(x)
        # print(result.shape)
        if self.training and self.gradloss > 0.:
            g = x['ysigma'][ok[...,0]]
            e = torch.ones_like(g, requires_grad=True)
            g = torch.autograd.grad((g*e).sum(), x['yval'], create_graph=True, retain_graph=True)[0]
            g = torch.autograd.grad(-g.clamp(max=0.).sum(), e, create_graph=True, retain_graph=True)[0]
            g = (g*g).sum(-1)
            loss[ok] += g*self.gradloss
            
            if self.embedding > 0:
                g = x['ysigma'][ok[...,0]]
                e = torch.ones_like(g, requires_grad=True)
                g = torch.autograd.grad((g*e).sum(), x['yemba'], create_graph=True, retain_graph=True)[0]
                g = torch.autograd.grad(-g.clamp(max=0.).sum(), e, create_graph=True, retain_graph=True)[0]
                g = (g*g).sum(-1)
                loss[ok] += g*self.gradloss
            # print([g.mean(), g.max()])
        if self.dist.returnsd and 'sd' in x:# and lossfun != 'mse':
            ok = torch.isfinite(x['target']) & torch.isfinite(x['sd']) & torch.isfinite(result) & (x['sd'] > 0.)
            pred = x['target'][ok] - result[ok]; sd = x.pop('sd')[ok].clamp(min=0.2,max=50.)
            # print(sd.min())
            nrm = (pred/sd).mul_(np.sqrt(0.5)).clamp(min=-6.,max=6.)
            # loss[ok] = (pred*torch.erf(nrm)).add_((torch.exp(-nrm*nrm-0.5*np.log(2.))-(sd/x['ysigma'][ok] if self.sigmed else sd)).mul_(np.sqrt(1./np.pi)))
            nrm = torch.exp(-nrm)*sd
            loss[ok] = ((pred+nrm)*(pred>0)+(sd+nrm)*0.5*(pred<0)).div_(np.sqrt(0.5))
            if self.usee and self.enet.returnsd and 'esd' in x and 'yval' in x:
                ok1 = torch.isfinite(x['target']) & torch.isfinite(x['esd']) & torch.isfinite(x['yval']) & (x['esd'] > 0.)
                pred = x['target'][ok1] - x['yval'][ok1]; sd = x.pop('esd')[ok1].clamp(min=0.2,max=50.)
                # print(sd.min())
                nrm = (pred/sd).mul_(np.sqrt(0.5)).clamp(min=-6.,max=6.)
                # loss[ok1] = loss[ok1] + 0.2*((pred*torch.erf(nrm)).add_((torch.exp(-nrm*nrm-0.5*np.log(2.))-(sd)).mul_(np.sqrt(1./np.pi))))
                nrm = torch.exp(-nrm)*sd
                loss[ok1] = loss[ok1] + ((pred+nrm)*(pred>0)+(sd+nrm)*0.5*(pred<0)).div_(5.*np.sqrt(0.5))
                cnt += 0.2*ok1.sum(dims).float()
            if lab+'_sd' not in self.history:
                self.history[lab+'_sd'] = []
            self.history[lab+'_sd'].append ((loss.sum()/cnt.sum()).item())
        if lab+'_n' not in self.history:
            self.history[lab+'_n'] = []
        self.history[lab+'_n'].append (cnt.sum().item())
        return loss.sum(dims), cnt

Model0 = DLOA(0,1,0,implicit=False,edging=True,rmax=1.,usee=False,biased=False,relativer=True).eval()
