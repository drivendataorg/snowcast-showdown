2022-02-10 - 2022-03-10
Initial submission

2022-03-17
Added visualization:
Changed scripts in the directory src/visualization
In main.py added lines 152-159
        figdir = path.join (maindir, 'reports', 'figures')
        makedirs (figdir,exist_ok=True)
        results = models.applymodels (inputs['test'], modelslist, average=False).clamp_(min=0.)
        # monests = visualization.monthests (inputs, dates, uregions, result)
        visualization.temporal(inputs['test'], dates['test'], results, result, uregions, figdir)
        visualization.spatial(inputs['test'], dates['test'], result, figdir)
        visualization.importance(inputs['test'], dates['test'], modelslist, result, arglist, nousest, figdir)
        visualization.latentsd(inputs['test'], days['test'], 8, modelslist[1], figdir)

2022-03-24 - 2022-06-30
Improved logging
In src/main.py added line 13
from traceback import format_exc
In src/main.py changed lines 131-154 (after writing submission file)
	print ('Mean values: ' + str({key: np.round(sub[key].mean(),4) for key in sub}))
        try:
            isubs = pd.DataFrame({'week': [iweek], 'day': [evalday], 'file': [file]})
            subslistfile = path.join(workdir,'submissionslist.csv')            
            if path.isfile(subslistfile):
                subslist = pd.read_csv(subslistfile)
                pd.concat((pd.read_csv(subslistfile).set_index(subslist.columns[0]), isubs)).to_csv(subslistfile)
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
In src/features/inputs.py added lines 28-29
	stswe_test = pd.read_csv(file)
        file = path.join(workdir,'ground_measures_features.csv')
        stswe_test.to_csv(file)