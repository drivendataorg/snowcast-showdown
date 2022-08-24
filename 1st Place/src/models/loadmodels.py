from os import path
import torch
import models

def loadmodels(loadstr, inputs, modelsdir, arglist, print=print, modelslist=None, totalbest=[100.]):
    better = modelslist is not None
    if not better:
        modelslist = {}    
    for fold in range(1,100):
        fname = path.join(modelsdir, loadstr+'_'+str(fold-1)+'.pt')
        # print(fname)
        if path.isfile(fname):
            if fold > len(totalbest):
                totalbest = totalbest + [100.]*(fold - len(totalbest))
            bw = torch.load(fname, map_location='cpu')
            kw = {key: bw[key] for key in bw if not isinstance(bw[key], torch.Tensor)}
            if better and 'best' in kw and kw['best']>totalbest[fold-1]:
                if print is not None:
                    print('skip')
                continue
            if print is not None:
                print(kw)
            # fname = path.join('models', loadstr+'_f'+str(fold-1)+'.pt')
            # if path.isfile(fname):
            #     bw = torch.load(fname, map_location='cpu')
            wg = {key: bw[key].to(models.calcdevice) for key in bw if key not in kw}        
            if 'best' in kw:
                totalbest[fold-1] = kw.pop('best')
            mode = list(inputs.keys())[0]
            if 'nets.0.dist.kf.rm' in wg:
                m = models.Model3(None, inputs[mode]['xinput'].shape[-1], inputs[mode]['xval'].shape[-1], **kw)
                # for mm in m.nets:
                #     mm.net[0].frozen = True
            else:
                m = models.Model(inputs[mode]['xinput'].shape[-1], inputs[mode]['xval'].shape[-1], **kw)
                # m.net[0].frozen = True
            # printshape(wg)
            # printshape(m.state_dict())
            m.load_state_dict(wg)
            m.to(device=models.calcdevice).eval()            
            # print(m.importance())
            modelslist[fold-1] = m
    if print is not None:
        imp = [modelslist[i].importance(arglist) for i in modelslist]
        print (imp)
        print ({key: [i[key] for i in imp] for key in imp[0]})
    return modelslist