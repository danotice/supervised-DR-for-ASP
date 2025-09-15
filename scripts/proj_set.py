import IS_class as ip
import IS_eval as ipe
from IS_class import InstanceSpace

import pandas as pd
import numpy as np
import pickle as pkl

import copy
import time

import sys

# cores = 30

# metapath = '../metadata/tsp.csv'
# outpath = '../Results/tsp/'

# selFeats = ['feature_Mean_StdDevDist','feature_FracDistinctDists_2Digits',
#  'feature_RectangularArea','feature_CoeffvarNormalised_nNNds',
#  'feature_RatioNodesNearEdgesPlane','feature_NumClusters']


###### 

def prep(metadata, selFeats, outpath, scaler):

    iSpace = InstanceSpace()
    iSpace.fromMetadata(metadata, scaler=scaler,best='Best',source='source')

    iSpace.splitData(test_size=0.2, random_state=1111,scale=True, stratified=True)
    train_ind = iSpace.split_data['Yb_train'].index
    test_ind = iSpace.split_data['Yb_test'].index

    # train_out = pd.DataFrame(
    #     np.hstack([iSpace.split_data[f'{d}_train'] for d in ['Y','X']]),
    #     index = train_ind, columns = iSpace.algorithms + iSpace.featureNames
    # )
    # train_out.insert(0, 'source', iSpace.source.loc[train_ind])
    # train_out.insert(0, 'best', iSpace.split_data['Yb_train'])

    # test_out = pd.DataFrame(
    #     np.hstack([iSpace.split_data[f'{d}_test'] for d in ['Y','X']]),
    #     index = test_ind, columns = iSpace.algorithms + iSpace.featureNames
    # )
    # test_out.insert(0, 'source', iSpace.source.loc[test_ind])
    # test_out.insert(0, 'best', iSpace.split_data['Yb_test'])

    # train_out.to_csv(outpath + 'processed_train.csv', index_label='instances')
    # test_out.to_csv(outpath + 'processed_test.csv', index_label='instances')

    ## save split metadata
    metadata.loc[train_ind].to_csv(outpath + 'metadata_train.csv', index_label='instances')
    metadata.loc[test_ind].to_csv(outpath + 'metadata_test.csv', index_label='instances')

    pd.DataFrame(selFeats, 
             index=[f'f{i}' for i in range(1,len(selFeats)+1)], 
             columns=['feature']
        ).T.to_csv(outpath + 'selected_features.csv', index_label='Row')
    
def proj_sel(metadata, selFeats, outpath, scaler, cores, da_file=''):
    iSpace = InstanceSpace()
    iSpace.fromMetadata(metadata, scaler=scaler,best='Best',source='source')
    iSpace.path = outpath
    
    iSpace.dropFeatures(selFeats)

    train_ind = list(pd.read_csv(outpath + 'metadata_train.csv')['instances'])
    test_ind = list(pd.read_csv(outpath + 'metadata_test.csv')['instances'])
    iSpace.splitData_known(train_ind, test_ind, scale=True)

    iSpace.doParallel(cores)
    #iSpace.doVerbose()

    if da_file != '':
        da_proj = pd.read_csv(outpath + da_file)
        da_list = da_proj['proj'].unique()
        for da in da_list:
            iSpace.addProj(da,da_proj.loc[da_proj['proj']==da])
        
    iSpace.PCA(n_components=2)

    iSpace.PILOT(n_components=2,mode='num', num_params={'seed':111, 'ntries':5})
    iSpace.PLS(n_components=2)
    iSpace.getRelativePerf(min=True)
    iSpace.PLS(mode='rel', n_components=2)

    ip.saveProjCoords(iSpace.PlotProj, outpath + 'proj_coords.csv',
                      extra_cols=['Best','group'])
    
    with open(outpath + 'iSpace.pkl', 'wb') as f:
        pkl.dump(iSpace, f)

def proj_to_is(ipath, outpath, proj_file):
    with open(outpath + ipath, 'rb') as f:
        iSpace = pkl.load(f)
    
    da_proj = pd.read_csv(outpath + proj_file)
    da_list = da_proj['proj'].unique()
    for da in da_list:
        iSpace.addProj(da,da_proj.loc[da_proj['proj']==da])        
    
    
    with open(outpath + 'iSpace.pkl', 'wb') as f:
        pkl.dump(iSpace, f)


def proj_eval(ipath, outpath):
    with open(outpath + ipath, 'rb') as f:
        iSpace = pkl.load(f)

    is_eval = ipe.InstanceSpaceEval(iSpace, split=True)
    is_eval.proj_evaluation()

    with open(outpath + 'proj_eval.pkl', 'wb') as f:
        pkl.dump(is_eval, f)

def pred_eval(metadata,selFeats,ipath,outpath, cores, pred_params=None,tie_lab=None):
    with open(outpath + ipath, 'rb') as f:
        iSpace = pkl.load(f)
    
    ## original features
    x_all = pd.DataFrame(pd.concat([iSpace.split_data['Yb_train'],iSpace.split_data['Yb_test']]))
    x_all['group'] = ['train']*len(iSpace.split_data['Yb_train'])+['test']*len(iSpace.split_data['Yb_test'])

    zcols = [f'Z{i+1}' for i in range(iSpace.m)]
    x_all = pd.concat([
        pd.DataFrame(np.vstack([iSpace.split_data['X_train'],iSpace.split_data['X_test']]), columns=zcols, index=x_all.index),
        x_all], axis=1)
    iSpace.addProj('All', x_all, 
                {'col_names':dict(zip(zcols,iSpace.featureNames))})

    iSpace.doParallel(cores)
    is_pred = ipe.PredictionEval(iSpace.PlotProj, split=True)

    metadata_train = pd.read_csv(f'{outpath}metadata_train.csv', index_col=0)
    pred_paramDF = ipe.pred_cv(metadata_train,selFeats,outpath,iSpace)

    if pred_params is None:
        pred_params = pd.DataFrame(
            {'SVM':[{}]*len(is_pred.projections.keys()),'KNN':[{}]*len(is_pred.projections.keys())},
            index=is_pred.projections.keys()
        )

    for proj in is_pred.projections.keys():
        is_pred.makePredictions_svm(proj, pred_paramDF.loc[proj,'SVM'])
        is_pred.makePredictions_knn(proj, pred_paramDF.loc[proj,'KNN'])
    
    is_pred.makePredictions_avg(avg_algo=metadata['Best'].value_counts().idxmax())
    
    for proj in is_pred.projections.keys():
        is_pred.evaluate_predictions(proj)
        is_pred.calc_regrets(proj,iSpace.performance,min=True,tie_lab=tie_lab)

    pd.DataFrame.from_dict(is_pred.evals, orient='index').reset_index().rename(
        columns={'level_0':'proj', 'level_1':'pred_model','level_2':'group'}
    ).to_csv(outpath + 'pred_eval.csv', index=False)

    with open(outpath + 'pred_eval.pkl', 'wb') as f:
        pkl.dump(is_pred, f)


def pred_eval_cv(metadata,selFeats,ipath,outpath, cores, tie_lab=None):
    with open(outpath + ipath, 'rb') as f:
        iSpace = pkl.load(f)
    
    ## original features
    x_all = pd.DataFrame(pd.concat([iSpace.split_data['Yb_train'],iSpace.split_data['Yb_test']]))
    x_all['group'] = ['train']*len(iSpace.split_data['Yb_train'])+['test']*len(iSpace.split_data['Yb_test'])

    zcols = [f'Z{i+1}' for i in range(iSpace.m)]
    x_all = pd.concat([
        pd.DataFrame(np.vstack([iSpace.split_data['X_train'],iSpace.split_data['X_test']]), columns=zcols, index=x_all.index),
        x_all], axis=1)
    iSpace.addProj('All', x_all, 
                {'col_names':dict(zip(zcols,iSpace.featureNames))})

    iSpace.doParallel(cores)
    is_pred = ipe.PredictionEval(iSpace.PlotProj, split=True)

    metadata_train = pd.read_csv(f'{outpath}metadata_train.csv', index_col=0)
    pred_paramDF = ipe.pred_cv(metadata_train,selFeats,outpath,iSpace)

    for proj in pred_paramDF.index:
        is_pred.makePredictions_svm(proj, pred_paramDF.loc[proj,'SVM'])
        is_pred.makePredictions_knn(proj, pred_paramDF.loc[proj,'KNN'])
    
    is_pred.makePredictions_avg(avg_algo=metadata['Best'].value_counts().idxmax())

    for proj in is_pred.projections.keys():
        is_pred.evaluate_predictions(proj)
        is_pred.calc_regrets(proj,iSpace.performance,min=True,tie_lab=tie_lab)

    pd.DataFrame.from_dict(is_pred.evals, orient='index').reset_index().rename(
        columns={'level_0':'proj', 'level_1':'pred_model','level_2':'group'}
    ).to_csv(outpath + 'pred_eval.csv', index=False)

    with open(outpath + 'pred_eval.pkl', 'wb') as f:
        pkl.dump(is_pred, f)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('metapath', type=str)
    parser.add_argument('outpath', type=str)
    # parser.add_argument('lab', type=str)
    parser.add_argument('step', type=int)
    parser.add_argument('cores', type=int)
    
    # optional argument scaler
    parser.add_argument('-scaler', type=str, default='s')
    parser.add_argument('-feat', type=str, default='selected_features.csv')
    parser.add_argument('-params', type=str, default='pred_params')
    parser.add_argument('-tie_lab', type=str, default=None)
    parser.add_argument('-rproj', type=str, default='da_proj.csv')

    args = parser.parse_args()

    metapath = args.metapath
    outpath = args.outpath
    scaler = args.scaler
    featS = args.feat
    cores = args.cores

    metadata = pd.read_csv(metapath, index_col=0)
    # metadata = pd.concat([
    #     pd.read_csv(f'{outpath}metadata_{t}.csv', index_col=0) for t in ['train','test']
    # ])

    if featS.endswith('.csv'):
        # selFeat_DF = pd.read_csv(f'../{featS}', index_col=0)
        # selFeats = list(selFeat_DF.loc[args.lab,'features'].split(' '))
        selFeats = list(pd.read_csv(
            outpath + 'selected_features.csv', index_col=0).iloc[0].values)

    else:
        print('Using all features')
        selFeats = [c for c in metadata.columns if c.startswith('feature_')]


    if args.step == 1:
        prep(metadata, selFeats, outpath, scaler)
    elif args.step == 2:
        proj_sel(metadata, selFeats, outpath, scaler, cores, da_file=args.rproj)
    elif args.step == 3:
        proj_eval('iSpace.pkl', outpath)    
    elif args.step == 4:
        pred_eval_cv(metadata, selFeats, 'iSpace.pkl', outpath, cores)

    elif args.step == 5:
        param_path = args.params
        if param_path.startswith('def'):
            pred_paramDF = None
        else:
            pred_paramDF = pd.read_csv(outpath+param_path+'.csv', index_col=0)

        pred_eval(metadata, None, 'iSpace.pkl', outpath, cores, 
                  pred_paramDF, args.tie_lab)
        
    elif args.step == 6:
        proj_to_is('iSpace.pkl', outpath, args.rproj)
        
    # update if successful
    f = open(outpath + 'status.txt', 'w')
    f.write('1')
    f.close()