import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime, timedelta
import numpy as np
from models import Model0
import os
import torch
import features

sind = lambda x: np.sin(np.radians(x))
cosd = lambda x: np.cos(np.radians(x))
def sundirect(data,lat,lon):
    D = data-datetime(2000,1,1)
    D = 2.*np.pi/365.242*(D.days+D.seconds/86400.)
    D = 0.39785*np.sin(D+np.radians(279.9348+1.9148*np.sin(D)-0.0795*np.cos(D)+0.0199*np.sin(D+D)-0.0016*np.cos(D+D)))
    D = np.sqrt(1.-D*D)+D*sind(lat)-1.
    cfi = cosd(lat)
    return {'CRayMin':D-cfi, 'CRayMax':D+cfi, 'SinYear':sind(lat)}
def printshape(a, prefix='', print=print):
    print (prefix+str({key: (a[key].dtype,a[key].shape,torch.isfinite(a[key]).sum().item() if isinstance(a[key],torch.Tensor) else np.isfinite(a[key]).sum()) for key in a} \
           if isinstance(a,dict) else [(x.dtype,x.shape,torch.isfinite(x).sum().item() if isinstance(x,torch.Tensor) else np.isfinite(x).sum()) for x in a]))

def getdatadict(rmode, stswe, gridswe, constfeatures, rsfeatures, maindevice, withpred=True, nmonths=1, print=print):
    # shifts = [1,2,4]
    shifts = [1,2,3,4]
    print ('getdatadict: '+(f'Withpred shifts: {shifts}' if withpred else 'Withoutpred'))
    if nmonths == 1:
        monthid = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 12:0}
    elif nmonths == 2:
        monthid = {1:0, 2:0, 3:1, 4:1, 5:1, 6:1, 12:0}
    elif nmonths == 3:
        monthid = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 12:0}
    elif nmonths == 6:
        monthid = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 12:6}    
    Model0m = Model0.to(maindevice)
    def apply0 (inputs, rmax=1., valid_bs=64):
        sh = inputs['ylo' if 'ylo' in inputs else 'xlo'].shape
        result = torch.zeros(sh+(inputs['xval'].shape[-1],), device=maindevice)
        batches = int(np.ceil(inputs['xlo'].shape[0]/valid_bs))    
        with torch.no_grad():
            for i in range (batches):
                sel = torch.arange(i*valid_bs, min((i+1)*valid_bs,inputs['xlo'].shape[0]), device=maindevice)
                result[sel] = Model0m ({key: inputs[key][sel] for key in inputs})
        return result
    days = {mode: features.getdays(stswe[mode]) for mode in stswe}    
    # if 'train' in days:
    #     days['train'] = [d for d in days['train'] if int(d[5:7])<=7]
    dates = {mode: [datetime.strptime(tday,'%Y-%m-%d') for tday in days[mode]] for mode in stswe}
    inputs = {mode: {} for mode in stswe}    
    for mode in stswe:
        for tday,date in zip(days[mode], dates[mode]):
            arg = {}
            for lab,d in zip('xy',[stswe, gridswe]):
                if tday in d[mode]:
                    val = [d[mode][tday].values]
                    if withpred:
                        pred = []
                        for dt in shifts:
                        # for dt in [1]:
                            tday1 = (date-timedelta(days=7*dt)).strftime('%Y-%m-%d')
                            pred.append(d[mode][tday1].values if tday1 in d[mode] else val[0])
                            bad = np.isnan(pred[-1])                            
                            pred[-1][bad] = val[0][bad] if dt == 1 else pred[-2][bad]
                        pred.reverse()
                        val += pred
                    arg.update({lab+'val': np.hstack([v[:,None] for v in val])})
                arg.update({lab+'lo': d[mode]['longitude'].values, lab+'la': d[mode]['latitude'].values, 
                            lab+'emb': d[mode]['emb'].values*nmonths+monthid[date.month]})
                s = sundirect(date, arg[lab+'la'], arg[lab+'lo'])
                arg[lab+'input'] = np.hstack( [d[mode][key].values[:,None] for key in constfeatures]+
                        [s[key][:,None] for key in s]+
                        [d[mode][tday+key].values[:,None] if tday+key in d[mode] else np.full((d[mode].shape[0],1),np.nan) for key in rsfeatures]
                       )
                arglist = constfeatures + list(s.keys()) + rsfeatures
            for key in arg:
                if key not in inputs[mode]:
                    inputs[mode][key] = []
                inputs[mode][key].append(arg[key][None])
        for key in inputs[mode]:
            x = np.vstack(inputs[mode][key])
            inputs[mode][key] = torch.tensor(x if key[1:] in ['lo', 'la', 'emb'] or x.ndim==3 else x[:,:,None], device=maindevice, 
                                             dtype=torch.long if key[1:]=='emb' else torch.float32)
        if mode == 'train' or rmode not in ['oper']:
            inputs[mode]['target'] = inputs[mode]['yval' if 'yval' in inputs[mode] else 'xval'][...,:1]         
    print(arglist)
    # xtoy = torch.tensor(np.array([a in ['CDEC','SNOTEL'] for a in arglist]),device=maindevice)
    # ytox = torch.tensor(np.array([a in uregions for a in arglist]),device=maindevice)
    with torch.no_grad():
        for mode in inputs:
            printshape(inputs[mode], prefix=mode+': ', print=print)
            if rmode == 'oper':
                x = {key: inputs[mode][key] for key in inputs[mode] if key[1:] in ['lo','la']}
                xx = {'x'+key: torch.cat((x['y'+key],x['x'+key]),1).detach() for key in ['lo','la']}
                for i, a in enumerate(arglist):
                    res1 = torch.cat((inputs[mode]['xinput'][...,i],inputs[mode]['yinput'][...,i]),1).detach()
                    bad = torch.isnan (res1)
                    ok = ~bad
                    if ((bad.sum(1)>0) & (ok.sum(1)>0)).any():
                        xx ['xval'] = res1[...,None].clone()
                        xy = {key: xx[key][:,ok.any(0)] for key in xx}
                        bad1 = bad.any(0)
                        xy.update ({'y'+key[1:]: xx[key][:,bad1] for key in xx})
                        print(f"{a}: ok={ok.sum().item()} bad={bad.sum().item()}")
                        # printshape(xy, print=print)
                        res2 = res1[:,bad1]
                        res2[bad[:,bad1]] = apply0 (xy)[...,0][bad[:,bad1]]
                        inputs[mode]['xinput'][...,i] = res1[:,:x['xlo'].shape[1]]
                        inputs[mode]['yinput'][...,i] = res1[:,x['xlo'].shape[1]:]
            # if xtoy.sum()>0:
            #         x = {key: inputs[mode][key] for key in inputs[mode] if key[1:] in ['lo','la']}
            #         x['xval'] = inputs[mode]['xinput'][..., xtoy]
            #         # x['xval'][torch.isnan(x['xval']).any(-1)] = np.nan
            #         inputs[mode]['yinput'][..., xtoy] = apply0 (x)
            # if ytox.sum()>0:
            #         x = {key: inputs[mode][('y' if key[0]=='x' else 'x')+key[1:]] for key in inputs[mode] if key[1:] in ['lo','la']}
            #         x['xval'] = inputs[mode]['yinput'][..., ytox]
            #         # x['xval'][torch.isnan(x['xval']).any(-1)] = np.nan
            #         inputs[mode]['xinput'][..., ytox] = apply0 (x)
    if 'train' in inputs:
        inputs['train']['istrainy'] = torch.isfinite(inputs['train']['yval'][...,0]).sum(0) > 0
        for key in inputs['train']:
            if key[0] == 'y' or key == 'target':
                inputs['train'][key] = inputs['train'][key][:,inputs['train']['istrainy']]
    print("getdatadict finish")
    return inputs, arglist, days, dates