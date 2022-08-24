import numpy as np
from os import path
import matplotlib.pyplot as plt
import torch
import models

meannan = lambda x,dim=1: torch.nan_to_num(x,0.).sum(dim)/torch.isfinite(x).sum(dim)
def temporal(inputs, dates, results, result, uregions, figdir):
    mon = np.array([d.month for d in dates])
    ticks = [dates[i] for i in (np.nonzero(mon[1:]>mon[:-1])[0]+1)]
    tickslabel = [d.strftime('%Y-%m-%D') for d in ticks]
    for snotonly in [True, False]:
        for ireg,reg in enumerate(uregions+['others', 'all']):
            if reg in uregions:
                region = inputs['yinput'][0,:,ireg]>0.5
            elif reg == 'others':
                region = inputs['yinput'][0,:,:2].sum(-1)<0.5
            else:
                region = torch.full(inputs['yinput'].shape[1:2], True)
            if snotonly:
                ok = torch.isfinite(inputs['target']).all(0)[:,0]&region
            else:
                ok = region
            print(reg+(' SNOTEL' if snotonly else ' ALL')+f' count={ok.sum().item()}')
            trg = inputs['target'][:,ok]
            e  = results[:,ok]-trg
            em = result[:,ok]-trg[...,0]
            e = torch.sqrt(meannan(e*e))
            fig, ax = plt.subplots(1,1)
            ax.set_title(reg+(f' {ok.sum().item()} locations' if snotonly else ''))
            ax.errorbar(dates, e.mean(-1).numpy(), e.std(-1).numpy(), label='one model error')
            ax.plot(dates, torch.sqrt(meannan(em*em)).numpy(), '--', label='mixed model error')
            ax.legend(loc='lower left')
            ax2 = ax.twinx()
            ax2.plot(dates, meannan(trg).numpy(), '-g', label='mean SWE (right axis)')
            ax2.legend(loc='upper right')
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickslabel)
            fig.savefig(path.join(figdir,'Errors'+('_SNOTEL' if snotonly else '_ALL')+reg+'.png'), dpi=300)

import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker, cm, colors
resoln = '10m'
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resoln, edgecolor='none', facecolor=cfeature.COLORS['water'])
# land = cfeature.NaturalEarthFeature('physical', 'land', scale=resoln, edgecolor='k', facecolor=cfeature.COLORS['land'])
lakes = cfeature.NaturalEarthFeature('physical', 'lakes', scale=resoln, edgecolor='b', facecolor=cfeature.COLORS['water'])
rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resoln, edgecolor='b', facecolor='none')
country_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resoln, facecolor='none', edgecolor='k')
provinc_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale=resoln, facecolor='none', edgecolor='k')
crs = ccrs.PlateCarree()
def back (lon1=-124, lon2=-104, lat1=32, lat2=50, grid=True):
    fig = plt.figure (figsize=(10, 10), edgecolor='w') 
    proj = ccrs.Stereographic (central_longitude=(lon1+lon2)*0.5, central_latitude=(lat1+lat2)*0.5)
    ax = fig.add_subplot (1, 1, 1, projection=proj)
    ax.set_extent ([lon1, lon2, lat1, lat2], crs=crs)
    ax.add_feature (ocean, linewidth=2., edgecolor='b', facecolor='none', zorder=14)
    ax.add_feature (lakes, linewidth=1., edgecolor='b', facecolor='b', zorder=14)
    ax.add_feature (rivers, linewidth=1., edgecolor='b', facecolor='none', zorder=14)
    ax.add_feature (country_borders, linewidth=2., edgecolor='k', facecolor='none', zorder=14)
    ax.add_feature (provinc_borders, linewidth=1., edgecolor='k', facecolor='none', zorder=14)
    if grid:
        gl = ax.gridlines (draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right = False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = gl.ylabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = gl.ylabel_style = {'color': 'k', 'weight': 'bold'}
    return fig, ax

def spatial(inputs, dates, result, figdir):
    snot = torch.isfinite(inputs['target']).all(0)[:,0]
    em = result[:,snot]-inputs['target'][:,snot,0]
    lon = inputs['ylo'][0,snot].numpy()
    lat = inputs['yla'][0,snot].numpy()
        
    for mon in range(6):            
        fig,ax = back ()
        bounds = np.arange(0.2,5.,0.2)
        cmap = cm.get_cmap ('viridis')
        norm = colors.BoundaryNorm (bounds, cmap.N)
        emm = em[torch.tensor([d.month-1==mon for d in dates])]
        emm = torch.sqrt(meannan(emm*emm,0)).numpy()
        img = ax.scatter(lon, lat, c=emm, cmap=cmap, norm=norm, transform=crs)
        plt.colorbar (img, fraction=0.017, pad=0.04, ticks=bounds[4::5])
        tit = ['January','February','March','April','May','June','July'][mon]
        ax.set_title (tit)#+f" mean RMSE={(emm*emm).mean():.3}")
        fig.savefig(path.join(figdir,tit+'.png'), dpi=300)

def importance(inputs, dates, modelslist, result, arglist, nousest, figdir):
     groups = {'region': lambda a: a in ['central rockies', 'sierras', 'CDEC'],
               'LandCover': lambda a: a[:9]=='GLOBCOVER',
               'soil': lambda a: a[:4]=='SOIL',
               'aspect': lambda a: a[:4] in ['aspe', 'east', 'sout'],
               'elevation': lambda a: a[:9]=='elevation',
               'MODIS': lambda a: a[:16]=='MNDSI_Snow_Cover',
               'regular': lambda a: a in ['isemb']}
     ret = {}
     def permutei (x, iperm):
         xperm = torch.randperm(x[...,0].numel())
         xin = x.clone()
         xin[..., iperm] = xin[..., iperm].reshape(xperm.shape[0],-1)[xperm].reshape(*x.shape[:2], -1)
         return xin
     for f in groups:
         iperm = torch.tensor ([groups[f](a) for a in arglist])
         inp = {key: permutei (inputs[key], iperm) if key[1:] == 'input' 
                 else inputs[key] for key in inputs}
         ret[f] = models.inference (inp, modelslist, dates, lab=f, print=print, nousest=nousest)
     e0 = result-inputs['target'][:,:,0]
     e0 = torch.sqrt(meannan((e0*e0).reshape(-1),0))
     imp = {}
     for f in groups:
         e = ret[f]-inputs['target'][:,:,0]
         e = torch.sqrt(meannan((e*e).reshape(-1),0))
         imp[f] = (e-e0).item()
     order = np.argsort([imp[f] for f in imp])[::-1]        
     keys = list(imp.keys())
     fig, ax = plt.subplots(1,1)
     ax.bar(['regular' if keys[o]=='ASO' else keys[o] for o in order], [imp[keys[o]] for o in order])
     fig.savefig(path.join(figdir,'Importance.png'), dpi=200)

def latentsd(inputs, days, k, model, figdir):
    x = {key: inputs[key][k:k+1].detach().to(models.calcdevice) for key in inputs if key[0] != 'i'}
    res,sd = model(x)
    
    val,vec=torch.eig((x['ymap'][0,:,:,None]*x['ymap'][0,:,None]).sum(0), eigenvectors=True)
    cv = {lab: (x[lab+'map'][0]*vec[:,0]).sum(-1).detach().cpu().numpy() for lab in 'xy'}
    
    cmap = cm.get_cmap ('viridis')
    for loc in ['', '_CA']:
        for visual in ['Spread', 'Latent']:
            if loc == '_CA':
                fig,ax = back (-122, -118, 36, 40.5)
            else:
                fig,ax = back ()
            if visual == 'Spread':
                bounds = np.arange(0.4,5.,0.2)*10.                
                norm = colors.BoundaryNorm (bounds, cmap.N)
                for lab in 'xy':
                    img = ax.scatter(x[lab+'lo'][0].cpu().numpy(), x[lab+'la'][0].cpu().numpy(),
                                      c=10.*x[lab+'sigma'][0,:,0].detach().cpu().numpy(), cmap=cmap, norm=norm, transform=crs)
                plt.colorbar (img, fraction=0.017, pad=0.04, ticks=bounds[3::5])
            elif visual == 'Latent':
                bounds = np.arange(0.,56.)
                norm = colors.BoundaryNorm (bounds, cmap.N)
                for lab in 'xy':
                    img = ax.scatter(x[lab+'lo'][0].cpu().numpy(), x[lab+'la'][0].cpu().numpy(),
                                      c=cv[lab]-cv['y'].min(), cmap=cmap, norm=norm, transform=crs)
                plt.colorbar (img, fraction=0.017, pad=0.04, ticks=bounds[::5])
            ax.set_title (f"{visual} {days[k]}")
            fig.savefig(path.join(figdir,f'{visual} {days[k]}{loc}.png'), dpi=300)
            