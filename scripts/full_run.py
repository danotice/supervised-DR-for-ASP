import IS_class as ip
from IS_class import InstanceSpace
import IS_eval as ipe
from proj_set import *

import pandas as pd
import numpy as np
import pickle as pkl


# input data
metadata = pd.read_csv('./metadata/tsp.csv', index_col=0)
outpath = './results/tsp_all/'

# using all features
selFeats = [c for c in metadata.columns if c.startswith('feature_')]

# creating instance space
iSpace = InstanceSpace()
iSpace.fromMetadata(metadata, scaler='s',best='Best',source='source')
iSpace.path = outpath
iSpace.dropFeatures(selFeats)

iSpace.splitData(test_size=0.2, random_state=1111,scale=True, stratified=True)
iSpace.PILOT(n_components=2,mode='a')

# adding all original features
x_all = pd.DataFrame(pd.concat([iSpace.split_data['Yb_train'],iSpace.split_data['Yb_test']]))
x_all['group'] = ['train']*len(iSpace.split_data['Yb_train'])+['test']*len(iSpace.split_data['Yb_test'])

zcols = [f'Z{i+1}' for i in range(iSpace.m)]
x_all = pd.concat([
    pd.DataFrame(np.vstack([iSpace.split_data['X_train'],iSpace.split_data['X_test']]), columns=zcols, index=x_all.index),
    x_all], axis=1)
iSpace.addProj('All', x_all, 
            {'col_names':dict(zip(zcols,iSpace.featureNames))})

# making predictions with default hyperparameters
is_pred = ipe.PredictionEval(iSpace.PlotProj, split=True)

is_pred.makePredictions_svm('PILOTa', params={})
is_pred.makePredictions_svm('All', params={})

is_pred.makePredictions_avg(avg_algo=metadata['Best'].value_counts().idxmax())
    
# prediction evaluation
for proj in is_pred.projections.keys():
    is_pred.evaluate_predictions(proj)
    is_pred.calc_regrets(proj,iSpace.performance,min=True,tie_lab=None)

# saving results
pd.DataFrame.from_dict(is_pred.evals, orient='index').reset_index().rename(
        columns={'level_0':'proj', 'level_1':'pred_model','level_2':'group'}
    ).to_csv(outpath + 'pred_eval.csv', index=False)

with open(outpath + 'pred_eval.pkl', 'wb') as f:
    pkl.dump(is_pred, f)

with open(outpath + 'iSpace.pkl', 'wb') as f:
    pkl.dump(iSpace, f)