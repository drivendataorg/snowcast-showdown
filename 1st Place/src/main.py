from argparse import ArgumentParser
import copy
from datetime import datetime, timedelta
import gc
import numpy as np
import logging
import pandas as pd
from os import path,listdir,makedirs
import matplotlib.pyplot as plt
from tictoc import TicTocGenerator
from threading import Thread, Lock
import torch
from traceback import format_exc
import features
import visualization
import models
from models import addons

awsurl = 'https://drivendata-public-assets.s3.amazonaws.com/'

parser = ArgumentParser(description='Snowcast Showdown solution (c) FBykov')
parser.add_argument('--mode', default = 'oper', type=str, help='mode: train/oper/test/finalize')
# parser.add_argument('--mode', default = 'train', type=str, help='mode: train/oper/test/finalize')
# parser.add_argument('--mode', default = 'finalize', type=str, help='mode: train/oper/test/finalize')
# parser.add_argument('--mode', default = 'test', type=str, help='mode: train/oper/test/finalize')
parser.add_argument('--maindir', default = '..', type=str, help='path to main directory')

#### Options for training - not significant for inference ####
parser.add_argument('--implicit', default = 0, type=int, help='implicit')
parser.add_argument('--individual', default = 0, type=int, help='individual')
parser.add_argument('--relativer', default = 0, type=int, help='relativer')
parser.add_argument('--norm', default = 0, type=int, help='normalization')
parser.add_argument('--springfine', default = 0, type=int, help='fine on spring')
parser.add_argument('--stack', default = 0, type=int, help='stack')
parser.add_argument('--calibr', default = 0, type=int, help='calibration')
parser.add_argument('--embedding', default = 0, type=int, help='embedding size')
parser.add_argument('--lossfun', default = 'smape', type=str, help='loss function')
parser.add_argument('--modelsize', default = '', type=str, help='modelsize')
#### Options for training - not significant for inference ####

try:
    args = parser.parse_args()
    args.notebook = False
except:
    args = parser.parse_args(args=[])
    args.notebook = True
args = args.__dict__
      
maindir = args['maindir']
datestr = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
logdir = path.join (maindir, 'logs')
makedirs (logdir,exist_ok=True)
logfile = path.join (logdir, datestr+'.log')
logging.basicConfig (level=logging.INFO, format="%(asctime)s %(message)s",
                     handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
print = logging.info
print(f"Logging to {logfile}")

print(f"Starting mode={args['mode']} maindir={maindir} args={args}")
workdir = path.join(maindir,'data','evaluation' if args['mode'] == 'oper' else 'development')
makedirs (workdir,exist_ok=True)
print(f"workdir={workdir} list={listdir(workdir)}")

modelsdir = path.join(maindir,'models')
maindevice = torch.device('cpu')   
withpred = True
nmonths = 1
print(f"arxiv on {maindevice} calc on {models.calcdevice}")
inputs, arglist, uregions, stswe, gridswe, days, dates, nstations = \
    features.getinputs(workdir, args['mode'], modelsdir, withpred=withpred, nmonths=nmonths, 
                       maindevice=maindevice, awsurl=awsurl, print=print)

gc.collect()
if torch.cuda.is_available():
    print ([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

nousest='isemb' not in arglist
modelslist={}
if args['mode'] in ['oper','test']:
    models.apply.valid_bs = 8
    lab0 = ('finalize' if args['mode'] in ['oper'] else 'train')
    totalbest=[100.]*8
    if args['mode'] in ['test']:
        spring = np.array([d.month*100+d.day for d in dates['test']])
        ok = torch.tensor(np.array([(d.year>=2021)|(d.month>10) for d in dates['test']]), device=maindevice)
        ok = ok&torch.tensor((spring>=215)&(spring<=701), device=maindevice)
        for key in inputs['test']:
            inputs['test'][key] = inputs['test'][key][ok]
        ok = ok.nonzero().cpu().numpy()[:,0]
        dates['test'] = [dates['test'][i] for i in ok]
        days['test'] = [days['test'][i] for i in ok]
    # for lab in [ '_ex_ind', '_imp_ind' ]:
    # for lab in [ '_ex_ind', 'a_ex_ind', '_imp_ind', 'a_imp_ind' ]: # 14
    # for lab in [ '_ex_ind_noemb', '_imp_ind_noemb', '_ex_com_noemb', '_imp_com_noemb',
    #              '_ex_ind', '_imp_ind', '_ex_com', '_imp_com']: #15 v1
    for lab in [ '_ex_ind_noemb', '_imp_com_noemb', '_ex_ind', '_imp_com']: #15 v2
        for big in ['B', '']:
            file = lab0+big+'_smapea'+lab
            try:
                models1 = models.loadmodels(file, inputs, modelsdir, arglist, print=None)
                print(f"Loaded {file} {len(models1)} models from {modelsdir}")
            except:
                print(f"Cannot load models {file}")
                continue
            if len(models1) == 0:
                print(f"Cannot load models {file}")
                continue
            for s in models1:
                if (big != 'B') or s not in [7,8]:
                    modelslist[len(modelslist)] = models1[s]
            # if args['mode'] in ['test']:
            #     for s in models1:
            #         r = models.inference (inputs['test'], {0: models1[s]}, dates['test'], lab=lab+str(s), print=print, nousest=nousest)
            #     r = models.inference (inputs['test'], models1, dates['test'], lab=lab, print=print, nousest=nousest)
    result = models.inference (inputs['test'], modelslist, dates['test'], print=print, nousest=nousest)
            
    if args['mode'] in ['oper']:        
        iweek = torch.isfinite(result).any(1).nonzero()[-1].item()+1
        evalday = days['test'][iweek-1]
        out = {tday: result[i] for i,tday in enumerate(days['test'])}
        out['cell_id'] = gridswe['test'].index
        sub = pd.DataFrame(out).set_index('cell_id')
        file = path.join(evalday,'submission'+datestr+'.csv')
        fname = path.join(workdir,file)
        subdir = path.join(workdir,evalday)
        makedirs (subdir,exist_ok=True)
        print('Saving submission to '+fname)
        sub.to_csv(fname)
        print ('Mean values: ' + str({key: np.round(sub[key].mean(),4) for key in sub}))
        try:
            isubs = pd.DataFrame({'week': [iweek], 'day': [evalday], 'file': [file]})
            subslistfile = path.join(workdir,'submissionslist.csv')            
            if path.isfile(subslistfile):
                subslist = pd.read_csv(subslistfile)
                pd.concat((subslist.set_index(subslist.columns[0]), isubs)).to_csv(subslistfile)
                isst = (inputs['test']['yemb'][0]>0).cpu().numpy()
                for file in subslist['file']:
                    try:
                        old = pd.read_csv(path.join(workdir,file)).set_index('cell_id')
                        print (f'Compare with {file}\n'+
                               str({key: [np.round((sub[key]-old[key])[isst].mean(),4), np.round(np.abs(sub[key]-old[key])[isst].mean(),4),
                                          np.round((sub[key]-old[key])[~isst].mean(),4), np.round(np.abs(sub[key]-old[key])[~isst].mean(),4)]
                                for key in old if np.isfinite(sub[key]).sum()>0}))
                    except:
                        print(format_exc())
            else:
                isubs.to_csv(subslistfile)
            from shutil import copyfile
            copyfile(path.join(workdir,'ground_measures_features.csv'), path.join(subdir,'ground_measures_features.csv'))
            copyfile(logfile, path.join(subdir,datestr+'.log'))
        except:
            print(format_exc())
    if args['mode'] in ['test']:
        figdir = path.join (maindir, 'reports', 'figures')
        makedirs (figdir,exist_ok=True)
        results = models.applymodels (inputs['test'], modelslist, average=False).clamp_(min=0.)
        # monests = visualization.monthests (inputs, dates, uregions, result)
        visualization.temporal(inputs['test'], dates['test'], results, result, uregions, figdir)
        visualization.spatial(inputs['test'], dates['test'], result, figdir)
        visualization.importance(inputs['test'], dates['test'], modelslist, result, arglist, nousest, figdir)
        visualization.latentsd(inputs['test'], days['test'], 8, modelslist[1], figdir)

elif 'train' in inputs:
    for key in ['relativer','implicit','individual','norm','springfine']:
        args[key] = args[key]>0.5
    inputs['train']['isstation'] = torch.isfinite(inputs['train']['yval'][...,0]).sum(0)>10        
    # visualization.boxplot(inputs,arglist)
    splits = np.array([True]+[dates['train'][i+1]-dates['train'][i]>timedelta(days=100) for i in range(len(dates['train'])-1)]+[True]).nonzero()[0]
    folds = list(range(1,len(splits)))
    totalbest = [100.]*max(folds)    
    devices = [torch.device('cuda:'+str(igpu)) for igpu in range(torch.cuda.device_count())]
    lock = Lock()
    prefix = args['mode']+args['modelsize'][:1]+'_'+args['lossfun']+('' if args['relativer'] else 'a')+('_imp' if args['implicit'] else '_ex')+\
            ('_ind' if args['individual'] else '_com')+('_n' if args['norm'] else '')+(f"_s{args['stack']}" if args['stack']>0 else '')+\
            ('_c' if args['calibr'] else '')+('_noemb' if args['embedding']==0 else '')+('_s' if args['springfine'] else '')
    # prefix = datestr
    
    def getinitemb(inputs, select, nstations, norm=True):
        pwr=0.6
        xv = inputs['xval'][select,:,0]
        device = xv.device
        ok = torch.isfinite(xv)
        xv = torch.nan_to_num(xv,0.)**pwr #; ok = xv>0
        if norm:
            nrm = (xv.sum(1,keepdim=True)/(xv>0).float().sum(1,keepdim=True).clamp_(min=1.)).clamp_(min=0.1)
            # nrm = (xv.sum(1,keepdim=True)/ok.float().sum(1,keepdim=True).clamp_(min=1.)).clamp_(min=0.1)
            xv = xv/nrm
        initemb = torch.zeros(nstations*nmonths, device=device)
        count   = torch.zeros(nstations*nmonths, device=device)
        initemb.scatter_add_(0, inputs['xemb'][select].reshape(-1), xv.reshape(-1))
        count.scatter_add_(0, inputs['xemb'][select].reshape(-1), ok.reshape(-1).float())
        xv = inputs['yval'][select,:,0]
        ok = torch.isfinite(xv)
        xv = torch.nan_to_num(xv,0.)**pwr #; ok = xv>0
        if norm:
            xv = xv/nrm
        initemb.scatter_add_(0, inputs['yemb'][select].reshape(-1), xv.reshape(-1))
        count.scatter_add_(0, inputs['yemb'][select].reshape(-1), ok.reshape(-1).float())        
        initemb.div_(count.clamp(min=1.))
        me = initemb[count>10].mean()
        initemb[count<=10] = me
        return (initemb-me)*0.2
    
    valid_bs = models.valid_bs
    def trainnew (inputs, modelslist, netarg,
                  lab='', seed=0, folds=1, lossfun = 'mse', lossval = 'mse', autoepochs=10, onlyauto=False, schedulerstep=2., lenplateau=2, igpu=0, springfine=True,
                  lr=0.01, weight_decay=0.001, bs=3, model3=1, freeze_ep=2, epochs=50, initdecay=0.8, L1=0., best=[100.], prune=1, pruneval=0.05, points0=3, fine=False):
        kw = copy.deepcopy(netarg)
        device = devices[igpu]
        arxdevice = inputs['train']['xlo'].device
        print(lab+f" arxiv on {arxdevice} calc on {device} save to {prefix}")
        TicToc = TicTocGenerator()
        mode = 'train'
        selecty= inputs[mode]['isstation'].to(device)
        if len(best) < max(folds):
            best = best*max(folds)
        for fold in folds:
            gc.collect()
            torch.cuda.empty_cache()
            if len(folds) > 1:
                subset = torch.cat((torch.arange(splits[fold-1], device=arxdevice),torch.arange(splits[fold], inputs[mode]['xinput'].shape[0], device=arxdevice)))
                if fine:
                    spring = np.array([dates[mode][i].month*100+dates[mode][i].day for i in subset.cpu().numpy()])
                    subset = subset[torch.tensor((spring>=215)&(spring<=701),device=subset.device)]
                valset = torch.arange(splits[fold-1], splits[fold], device=arxdevice)
                spring = np.array([d.month*100+d.day for d in dates[mode][splits[fold-1]:splits[fold]]])
                valset = valset[torch.tensor((spring>=215)&(spring<=701),device=valset.device)]
                if ((torch.isfinite(inputs[mode]['xinput'][valset]).all(-1).sum(1)>kw['points']).sum() == 0) or \
                   ((torch.isfinite(inputs[mode]['xinput'][subset]).all(-1).sum(1)>kw['points']).sum() == 0):
                    continue
            better = False
            addons.set_seed(fold+int(100.*(kw['rmaxe']*(1. if kw['rmax'] is None else kw['rmax'])))+seed*1000)
            print(kw)            
            batches = int(np.ceil(inputs['train']['xinput'].shape[0]/bs))
            if fine:
                m = modelslist[fold-1].to(device=device)
                models.inference (inputs['test'], {0:m}, dates['test'], lab=lab+f' y={fold-1}_00', smart=True, device=device, print=print, nousest=nousest)
            else:
                netargin = [inputs[mode]['xinput'].shape[-1], inputs[mode]['xval'].shape[-1]]
                m = models.Model(*netargin, initemb=getinitemb(inputs[mode], subset, netarg['nstations']), **kw).to(device=device)
                # if model3 > 1:
                #     m = models.Model3(m.state_dict(), *netargin, layers=model3, **kw).to(device=device)
            decay = min(0.9,max(weight_decay,(1.0-np.power(initdecay, 1./batches/freeze_ep))/lr))
            print(lab+f' initdecay={decay} inputs={inputs[mode]["xinput"].shape[-1]} parameters={sum([p.numel() for p in m.parameters() if p.requires_grad])}')
            getoptimizer = lambda m, lr=lr, warmup=5: addons.AdamW(m.netparams(), weight_decay=decay, lr=lr, warmup=5, belief=True, gc=0., amsgrad=True, L1=L1)
            getkfoptimizer = lambda m, lr=lr: addons.AdamW(m.kfparams(), weight_decay=0., lr=lr, warmup=0, belief=False, gc=0., amsgrad=True)
            getscheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(0.1,1.0/(1.0 + step/schedulerstep)) )
            optimizer = getoptimizer (m)
            kfoptimizer = getkfoptimizer (m, lr=0.1*lr)
            scheduler = getscheduler (optimizer)
                    
            next(TicToc)
            cpoints = points0
            plateau = 0
            if prune > 0:
                nzero = m.nonzero()
                if fine:
                    for p in nzero:
                        w = torch.abs (p.data)
                        nzero[p] = w > torch.maximum(w.max(0,keepdim=True)[0],w.max(1,keepdim=True)[0]).clamp_(max=1.)*pruneval
            for epoch in range(epochs):                
                # frozen = min(2,(2*epoch)//autoepochs)
                frozen = min(2,epoch//freeze_ep)
                m.freeze(frozen)
                if epoch==freeze_ep and not fine:
                    for group in optimizer.param_groups:
                        group['weight_decay'] = weight_decay
                    if springfine:                        
                        spring = np.array([dates[mode][i].month*100+dates[mode][i].day for i in subset.cpu().numpy()])
                        subset = subset[torch.tensor((spring>=215)&(spring<=701),device=subset.device)]
                perm = subset[torch.randperm(subset.shape[0], device=arxdevice)]
                batches = int(np.ceil(perm.shape[0]/bs))
                if epoch > 0 and epoch*prune//freeze_ep <= prune and (epoch*prune)%freeze_ep < prune:
                    msg = ''
                    for p in nzero:
                        w = torch.abs (p.data)
                        nzero[p] = w > torch.maximum(w.max(0,keepdim=True)[0],w.max(1,keepdim=True)[0]).clamp_(max=1.)*pruneval
                        p.data.mul_(nzero[p])
                        if 'exp_avg' in optimizer.state[p]:
                            optimizer.state[p]['exp_avg'].mul_(nzero[p])
                        msg += f'{nzero[p].sum()}/{nzero[p].numel()} '
                    print(lab+' Nonzero grad '+msg)
                prune1 = prune if prune>0 else 3
                if model3>1 and epoch*prune1//freeze_ep == 1 and (epoch*prune1)%freeze_ep < prune1 and not fine:
                # if model3>1 and epoch == 0 and not fine:
                    m = models.Model3(m.state_dict(), *netargin, layers=model3, **kw).to(device=device)
                    if prune > 0:
                        nzero = m.nonzero (nzero)
                    print(lab+f' Model3 style={m.style} parameters={sum([p.numel() for p in m.parameters() if p.requires_grad])}')
                    optimizer = getoptimizer (m, lr=0.5*lr, warmup=batches//2)
                    kfoptimizer = getkfoptimizer (m, lr=0.5*lr)
                    scheduler = getscheduler (optimizer)
                    cpoints = points0
                if epoch == autoepochs and not onlyauto:
                    for opt in [optimizer, kfoptimizer]:
                        for p in opt.state:                                
                            if 'exp_avg' in opt.state[p]:
                                # opt.state[p]['step'] = 0
                                opt.state[p]['exp_avg'].fill_(0.)
                                # opt.state[p]['exp_avg_sq'].fill_(0.)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                        # group['L1'] /= 5.
                    scheduler = getscheduler(optimizer)
                    cpoints = points0+2
                sloss = 0.; cnts = 0.
                m.train()
                for i in range (batches):
                    m.points = cpoints+np.random.randint(-1,2)
                    sel = perm[i*bs:min((i+1)*bs,perm.shape[0])]
                    optimizer.zero_grad()
                    kfoptimizer.zero_grad()
                    res,loss,cnt = models.apply(m, inputs[mode], sel, device=device, 
                                                autoloss=(epoch<autoepochs) or onlyauto, lab=mode, lossfun=lossfun, selecty=selecty)
                    sloss += loss.item(); cnts += cnt.item()
                    (loss/cnt).mean().backward()
                    if epoch*prune1 >= freeze_ep:
                    # if epoch >= freeze_ep:
                        kfoptimizer.step()
                    if prune > 0 and epoch*max(1,prune) >= freeze_ep:
                        for p in nzero:
                            p.grad.data.mul_(nzero[p])
                    optimizer.step()
                sloss = np.sqrt(sloss/cnts) if lossfun=='mse' else sloss/cnts
                out = f"train={sloss:.4}"
                if len(folds) > 1:
                    m.eval()
                    m.points = kw['points']
                    batches = int(np.ceil(valset.shape[0]/valid_bs))
                    sloss = 0.; cnts = 0.
                    with torch.no_grad():
                        for i in range (batches):                                
                            sel = valset[torch.arange(i*valid_bs, min((i+1)*valid_bs,valset.shape[0]), device=arxdevice)]                                
                            res,loss,cnt = models.apply(m, inputs['train'], sel, device=device, lab='valid', lossfun=lossval, selecty=selecty)
                            sloss += loss.item(); cnts += cnt.item()
                    val = np.sqrt(sloss/cnts) if lossval=='mse' else sloss/cnts
                    out = f"valid={val:.4} " + out
                    if val < best[fold-1]:
                        best[fold-1] = val
                        best_weights = m.state_dict()
                        best_weights = {key: best_weights[key].clone() for key in best_weights}
                                                
                        plateau = 0
                        if epoch > autoepochs:
                            better = True
                        out = ('I best ' if totalbest[fold-1] > best[fold-1] else 'better ')+out
                        # if epoch*max(1,prune) >= autoepochs and 'test' in inputs:
#                        if (epoch >= freeze_ep or fine) and 'test' in inputs:
#                            models.inference (inputs['test'], {0:m}, dates['test'], lab=lab+f' y={fold-1}_{epoch}', device=device, print=print, nousest=nousest)
                    else:
                        out = '       '+out
                        plateau += 1
                        if plateau >= lenplateau:
                            if plateau > 20 and epoch > 2*autoepochs:
                                break
                            scheduler.step()
                            if plateau % 2 == 0:
                                cpoints = min(cpoints+1,kw['points'])
                print(lab + f" {epoch} " + out + f" {next(TicToc):.3}s lr={optimizer.param_groups[0]['lr']:.4}" + m.prints())
            if better:
                m.eval()
                m.points = kw['points']
                m.load_state_dict(best_weights)
                print(m.importance(arglist))
                if 'test' in inputs:
                    if 'target' in inputs['test']:
                        models.test(inputs, {0:m}, dates, lab, f' y={fold-1}', device=device, print=print, nousest=nousest)
                if args['mode'] == 'train' or epoch >= freeze_ep:
                    with lock:
                        if totalbest[fold-1] > best[fold-1] or (fold-1 not in modelslist):
                            # m.dist.graph()
                            best_weights.update(kw)
                            best_weights['best'] = best[fold-1]                            
                            torch.save(best_weights, path.join(modelsdir, prefix+'_'+str(fold-1)+'.pt'))
                            print(f'Copy {lab} to modelslist {fold-1} loss={best[fold-1]}'+m.prints())
                            modelslist[fold-1] = m
                            totalbest[fold-1] = best[fold-1]
        print(lab+f'\n {kw} \n best={best}')
        if 'test' in inputs:
            if 'target' in inputs['test']:
                models.test(inputs, modelslist, dates, lab, device=device, print=print, nousest=nousest)
        return modelslist
    
    netarg = {'usee': True, 'biased': False, 'sigmed': True, 'densed': True, 'implicit': args['implicit'],
              'edging': True,  'initgamma': None, 'norm': args['norm'], 'eps': -1.3, 'individual': args['individual'],
              'nf': [24], 'nlatent': 3,
#               'nf': [32], 'nlatent': 5,
              'embedding': args['embedding'], 'nmonths': nmonths, 'points': 20, 'gradloss': 0., 'style': 'stack',
              'nstations': nstations, 'dropout': 0.2, 'calcsd': 0, 'mc': 3,
              'rmaxe': 0.9, 'rmax': 0.7, 'rmax2': 0.5, 'relativer': args['relativer'], 'commonnet': True, 'calibr': args['calibr'] }
    ep = int(np.log (len(days['train'])/8))
    trarg = {'weight_decay': 0.01, 'L1': 0.01, 'initdecay': 0.35, 'igpu': 0,
              'best': [100.], 'lab': '', 'pruneval': 0.05, 'folds': folds, 'onlyauto': False,
              'lossfun': 'smape', 'lossval': 'mse', 'fine': False, 'springfine': args['springfine'],
              'lr': 0.01, 'bs': 2*ep*ep, 'freeze_ep': 10*ep, 'autoepochs': 15*ep,
              'lenplateau': 1, 'schedulerstep': 4.*ep, 'epochs': 70*ep, 'prune': ep, 'model3': 1+args['stack']}
    if netarg['mc']>1 and trarg['model3']>1:
        trarg['bs'] //= 2
        trarg['lr'] /= 2
    if True:
        threads=[None]*len(devices)
        igpu = 0        
        m3 = 1 #+withpred
        idec = 0.35
        for seed in range(3):
            for rmax in [0.9,0.7]:
#            for idec in [0.7,0.9,0.5,0.35]:
#                rmax = 0.8
                    rmaxe = np.round(rmax*13.)/10
                    nlatent = 3 if netarg['embedding']>0 else 5
                    if args['modelsize'][:1] == 'B':
                        nf = [48 if netarg['embedding']>0 else 96]
                    else:
                        nf = [24 if netarg['embedding']>0 else 48]
                # for nlatent,nf in zip([3 if netarg['embedding']>0 else 5],[[24 if netarg['embedding']>0 else 48]]):
#                for nlatent,nf in zip([5,3,2,5,3,2,3,2],[[16],[16],[16],[24],[24],[24],[32],[32]]):
                    trarg1 = copy.deepcopy(trarg)
                    netarg1 = copy.deepcopy(netarg)
                    igpu = (igpu+1)%len(devices)
                    netarg1.update ({'rmax': rmax, 'rmax2': rmax, 'rmaxe': rmaxe, 'nf': nf, 'nlatent': nlatent})
                    trarg1.update ({'lab': f"r={rmax}"+('r' if netarg1['relativer'] else '')+f",{rmaxe}nf={nf}nl={nlatent}_{args['lossfun']} "+
                                    ('imp' if netarg1['implicit'] else 'ex')+('n' if netarg1['norm'] else ' ')+f' m3={m3} idec={idec}',
                                    'igpu': igpu, 'lossfun': args['lossfun'], 'seed': seed, 'model3': m3, 'initdecay': idec})
                    if len(devices) > 1:
                        if threads[igpu] is not None:
                            threads[igpu].join()
                        print(trarg1)
                        threads[igpu] = Thread(target=trainnew, args=(inputs, modelslist, netarg1), kwargs=trarg1)
                        threads[igpu].start()
                    else:
                        print(trarg1)
                        modelslist = trainnew(inputs, modelslist, netarg1, **trarg1)
        for thread in threads:
            thread.join()        
    modelslist = models.loadmodels(prefix, inputs, modelsdir, arglist)
    
    print('valid='+np.array2string(np.array(totalbest)))
    imp = [modelslist[m].importance(arglist) for m in modelslist]
    for key in imp[0]:
        print({key: list([i[key] for i in imp])})
    if 'test' in inputs:
        r = models.test(inputs, modelslist, dates, print=print, nousest=nousest)
