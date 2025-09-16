## version 1.2
import pandas as pd
import numpy as np
import pickle as pkl
import os
from copy import copy, deepcopy

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition, cross_decomposition
from scipy.spatial import distance
from scipy import stats
from scipy.linalg import eig

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate, train_test_split

from projector_AE import Projector
from pilot import Pilot

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('mode.chained_assignment',None)

    
class IScaler():

    def __init__(self, scale_type):
        if scale_type == 's':
            self.scaler = StandardScaler()
        elif scale_type == 'r5':
            self.scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True)
        elif scale_type == 'r25':
            self.scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True)
        elif scale_type == 'p':
            self.scaler = PowerTransformer()
        else:
            self.scaler = None

    def fit(self, data):
        if self.scaler is not None:
            self.scaler.fit(data)
        return self
    
    def transform(self, data):
        if self.scaler is not None:
            return self.scaler.transform(data)
        return []
    
# TODO - 
# 1. raise flag if NaN or Inf in data
# 2. allow fromMetadata to take source
class InstanceSpace():
    
    def __init__(self):
        
        # attributes
        self.features = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.featureNames = []
        self.algorithms = []
        self.source = []

        self.n, self.m, self.a = 0, 0, 0 

        # standardized arrays
        self.X_s = []
        self.Y_s = []
        self.Yt_s = []

        # centered arrays
        # self.X_c = []
        # self.Y_c = []
        

        # projection spaces
        self.PlotProj = {} # projection of data points - 2D
        self.projections = {}   # projection objects
        
        # training-test split
        self.split_data = {}
        self.Y_rel = []

        # eval
        self.eval = {}
        self.footprints = {}

        # scaler
        self.scaler = None
        self.x_scaler = None
        self.y_scaler = None
        self.yt_scaler = None

        # verbose
        self.verbose = False
        self.parallel = False
        self.path = ''

    def fromMetadata(self, metadata, prefixes=['feature_','algo_'], scaler='s', best=None, source=None):
        metadata.index.name = 'instances'
        self.featureNames = [x for x in metadata.columns if x.startswith(prefixes[0])]
        self.algorithms = [x for x in metadata.columns if x.startswith(prefixes[1])]

        self.features = metadata[self.featureNames]
        self.performance = metadata[self.algorithms]

        if best is not None and best in metadata.columns:
            self.performance['Best'] = metadata[best]
        if source is not None and source in metadata.columns:
            self.source = metadata[source]

        # put instance name in PlotProj
        #self.PlotProj['instance'] = metadata.index
        #self.PlotProj.set_index(metadata.index,inplace=True)

        self.n, self.m = self.features.shape
        self.a = len(self.algorithms)

        self.scaler = scaler
        self.x_scaler = IScaler(scaler).fit(self.features.values)
        self.y_scaler = IScaler(scaler).fit(self.performance[self.algorithms].values)
        self.yt_scaler = IScaler(scaler).fit(self.performance[self.algorithms].values.reshape(-1,1))

        self.X_s = self.x_scaler.transform(self.features.values)
        self.Y_s = self.y_scaler.transform(self.performance[self.algorithms].values)
        self.Yt_s = np.apply_along_axis(
            lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.performance[self.algorithms].values).reshape(-1,self.a)


        # if scaler == 's':
        #     self.x_scaler = StandardScaler().fit(self.features.values)
        #     self.y_scaler = StandardScaler().fit(self.performance[self.algorithms].values)
        #     self.yt_scaler = StandardScaler().fit(self.performance[self.algorithms].values.reshape(-1,1))            
            
        # elif scaler == 'r5':
        #     self.x_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(self.features.values)
        #     self.y_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(self.performance[self.algorithms].values)
        #     self.yt_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True
        #                                  ).fit(self.performance[self.algorithms].values.reshape(-1,1))            
        
        # elif scaler == 'r25':
        #     self.x_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(self.features.values)
        #     self.y_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(self.performance[self.algorithms].values)
        #     self.yt_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True
        #                                  ).fit(self.performance[self.algorithms].values.reshape(-1,1))

        # elif scaler == 'p':
        #     self.x_scaler = PowerTransformer().fit(self.features.values)
        #     self.y_scaler = PowerTransformer().fit(self.performance[self.algorithms].values)
        #     self.yt_scaler = PowerTransformer().fit(self.performance[self.algorithms].values.reshape(-1,1))            

        # if self.x_scaler is not None:
        #     self.X_s = self.x_scaler.transform(self.features.values)
        #     self.Y_s = self.y_scaler.transform(self.performance[self.algorithms].values)
        #     self.Yt_s = np.apply_along_axis(
        #         lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.performance[self.algorithms].values).reshape(-1,self.a)
            
        
        # self.X_c = StandardScaler(with_std=False).fit_transform(self.features.values)
        # self.Y_c = StandardScaler(with_std=False).fit_transform(self.performance[self.algorithms].values)

    def getSource(self, source):
        if callable(source):
            self.source = source(self.features)
            self.PlotProj['Source'] = self.source
            print('source of data available')

        elif len(source) == self.n:
            self.source = source
            self.PlotProj['Source'] = self.source
            print('source of data available')

        else:
            print('Cannot assign source. Check the length of the list or the function')

    def getBest(self, best, axis=1):
        """Actual best performance based on performance metrics

        Args:
            expr: function with rule for best algorithm or a list type with the same length as the number of instances 
        """
        if callable(best):
            self.performance['Best'] = self.performance.apply(best,axis=axis)
            # self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        elif len(best) == self.n:
            self.performance['Best'] = best
            # self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        else:
            print('Cannot assign best algorithm. Check the length of the list or the function')
    
    def getBinaryPerf(self, abs, min, thr, pref='algo_'):
        if abs:
            if min:
                binPerf = self.performance[self.algorithms].apply(
                    lambda row: row <= thr, axis=1)
            else:
                binPerf = self.performance[self.algorithms].apply(
                    lambda row: row >= thr, axis=1)
        else:
            if min:
                binPerf = self.Y_rel.apply(
                    lambda row: row <= thr, axis=1)
            else:
                binPerf = self.Y_rel.apply(
                    lambda row: row >= thr, axis=1)
                
        # rename columns
        binPerf.rename(columns = {c: 'bin_'+c.removeprefix(pref) for c in binPerf.columns}, inplace=True)
        return binPerf

    def getRelativePerf(self, min):

        if min:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: row if row.min()==0 else np.abs((row - row.min())/row.min()), axis=1
            ).fillna(0)#.values
        else:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: row if row.max()==0 else np.abs((row - row.max())/row.max()), axis=1
            ).fillna(0)#.values       
        
        if self.split_data != {}:
            self.split_data['Yr_train'] = self.Y_rel.loc[self.split_data['Yb_train'].index]
            self.split_data['Yr_test'] = self.Y_rel.loc[self.split_data['Yb_test'].index]

        if self.verbose:
            print(f'Relative performance data available')

    def splitData(self, test_size, random_state, scale, stratified = True):
        """Split data into training and test sets.

        Args:
            test_size (float): proportion of data to be in test set
            random_state (int): seed for random number generator
            scale (bool): whether to scale the data
            stratified (bool, optional): whether to stratify the split. Defaults to True.
        """


        self.split_data = dict(zip(
            ['X_train', 'X_test', 'Y_train', 'Y_test','Yb_train', 'Yb_test'],
            
            train_test_split(self.features.values, self.performance[self.algorithms].values, 
                             self.performance['Best'],
                             test_size=test_size, random_state=random_state, 
                             stratify= self.performance['Best'] if stratified else None)
    
        ))

        if scale:
            self.x_scaler = IScaler(self.scaler).fit(self.split_data['X_train'])
            self.y_scaler = IScaler(self.scaler).fit(self.split_data['Y_train'])
            self.yt_scaler = IScaler(self.scaler).fit(self.split_data['Y_train'].reshape(-1,1))
                        
            self.split_data['X_train'] = self.x_scaler.transform(self.split_data['X_train'])
            self.split_data['X_test'] = self.x_scaler.transform(self.split_data['X_test'])
            self.split_data['Y_train'] = self.y_scaler.transform(self.split_data['Y_train'])
            self.split_data['Y_test'] = self.y_scaler.transform(self.split_data['Y_test'])
            self.split_data['Yt_train'] = np.apply_along_axis(
                lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_train']).reshape(-1,self.a)
            self.split_data['Yt_test'] = np.apply_along_axis(
                lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_test']).reshape(-1,self.a)
        
        else:
            self.split_data['X_train'] = self.split_data['X_train'].astype(float)
            self.split_data['X_test'] = self.split_data['X_test'].astype(float)

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'], self.split_data['Yr_test'] = train_test_split(
                self.Y_rel.values, test_size=test_size, random_state=random_state, 
                stratify= self.performance['Best'] if stratified else None
            )

        if self.verbose:
            ps = 'Scaled' if scale else 'Unscaled'
            print(f'{ps} data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets, stratified: {stratified}')
   
    def splitData_known(self, train_ind, test_ind, scale):

        self.split_data = {
            'X_train': self.features.loc[train_ind,:].values,
            'X_test': self.features.loc[test_ind,:].values,
            'Y_train': self.performance.loc[train_ind,self.algorithms].values,
            'Y_test': self.performance.loc[test_ind,self.algorithms].values,
            'Yb_train': self.performance.loc[train_ind,'Best'],
            'Yb_test': self.performance.loc[test_ind,'Best']
        }


        if scale:
            self.x_scaler = IScaler(self.scaler).fit(self.split_data['X_train'])
            self.y_scaler = IScaler(self.scaler).fit(self.split_data['Y_train'])
            self.yt_scaler = IScaler(self.scaler).fit(self.split_data['Y_train'].reshape(-1,1))
                        
            self.split_data['X_train'] = self.x_scaler.transform(self.split_data['X_train'])
            self.split_data['X_test'] = self.x_scaler.transform(self.split_data['X_test'])
            self.split_data['Y_train'] = self.y_scaler.transform(self.split_data['Y_train'])
            self.split_data['Y_test'] = self.y_scaler.transform(self.split_data['Y_test'])
            self.split_data['Yt_train'] = np.apply_along_axis(
                lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_train']).reshape(-1,self.a)
            self.split_data['Yt_test'] = np.apply_along_axis(
                lambda x: self.yt_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_test']).reshape(-1,self.a)
        
        else:
            self.split_data['X_train'] = self.split_data['X_train'].astype(float)
            self.split_data['X_test'] = self.split_data['X_test'].astype(float)

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'] = self.Y_rel.loc[train_ind].values
            self.split_data['Yr_test'] = self.Y_rel.loc[test_ind].values

        if self.verbose:
            ps = 'Scaled' if scale else 'Unscaled'
            print(f'{ps} data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets')
   
    def dropFeatures(self, keep):
        keep_ind = [self.featureNames.index(f) for f in keep]
        self.m = len(keep)

        self.featureNames = keep
        self.features = self.features.loc[:,keep]
            
        self.X_s = self.X_s[:,keep_ind]
        # self.X_c = self.X_c[:,keep_ind]
        
        if len(self.split_data) > 0:
            self.split_data['X_train'] = self.split_data['X_train'][:,keep_ind]
            self.split_data['X_test'] = self.split_data['X_test'][:,keep_ind]

        train_ind = self.split_data['Yb_train'].index if len(self.split_data) > 0 else self.performance.index

        self.x_scaler = IScaler(self.scaler).fit(self.features.loc[train_ind,:].values)          

        print(f'Features dropped. Remaining features: {self.m}')        

    def initPlotData(self, type='perf',feats=[]):
        """Puts all the data needed for plots in Proj dataframe.

        Args:
            type (str, optional): which data matrix to add (best, perf, feat). Defaults to 'best'.
        """

        if type=="perf":
            self.PlotProj[self.algorithms] = self.performance[self.algorithms]
            print('performance data available for visualisation')
        
        elif type=="feat":
            skipped = []
            for f in feats:
                if f in self.featureNames:
                    self.PlotProj[f] = self.features[f]
                else:
                    skipped.append(f)
            if len(skipped) > 0:
                print(f"Features {skipped} not found in metadata")
            print('feature data available for visualisation')
        
        else:
            print('No data added to PlotProj')

    def plot(self, proj, hue=None, legend=True):

        pltData = self.PlotProj[proj].copy()

        split = None if 'group' not in pltData.columns else 'group'

        if hue in pltData.columns:
            hue = hue
        elif hue == 'Best':
            if split == 'group':
                pltData['Best'] = pd.concat([
                    self.split_data['Yb_train'], self.split_data['Yb_test']
                ])
            else:
                pltData['Best'] = self.performance['Best']
        else:
            hue = None

        if len([col for col in pltData.columns if col.startswith('Z')]) == 1:
            # density plot - group by split            
            sns.kdeplot(x='Z1',hue=hue, data=pltData, fill=True, legend=legend)
        # or if Z2 is na
        elif pltData['Z2'].isna().all():
            sns.kdeplot(x='Z1',hue=hue, data=pltData, fill=True, legend=legend)   
        else: 
            sns.scatterplot(data=pltData, x='Z1', y='Z2', alpha=0.7, s=20,
                hue=hue, style=split, legend=legend)

    def doVerbose(self, verbose=True):
        self.verbose = verbose

    def doParallel(self, nodes):
        if nodes > 1 and isinstance(nodes, int):
            self.parallel = nodes
            print(f'Parallel mode set with {nodes} nodes')
        else:
            self.parallel = False
            print('Parallel mode not set')

    # adding projections
    def addProj(self, proj, proj_data, proj_obj=None):
        print(f'Manually adding {proj} projection')

        if 'proj' in proj_data.columns:
            proj_data.drop(columns='proj',inplace=True)
        if 'instances' in proj_data.columns:
            proj_data.set_index('instances',inplace=True)
            
        self.PlotProj[proj] = proj_data
        self.projections[proj] = proj_obj

    def PCA(self, hd=False, split=True, n_components=None):
        """Fit PCA and add projection to PlotProj. 
            Fit on scaled and centered data. 

        Args:
            n_components : number of components to keep. If None, all components are kept.
                If int, specifies the number of components to keep.
                If float, it specifies the cumulative explained variance.
        """

        if 'PCA' in self.projections.keys():
            print("PCA projection already added")
            return
        
        if n_components==None and not hd:
            n_components = 2
        
        pca = decomposition.PCA(n_components=n_components, random_state=11)
            
        if split:
            pca.fit(self.split_data['X_train'])
            comps = [f'Z{i+1}' for i in range(pca.n_components_)]
        
            df_train = pd.DataFrame(
                pca.transform(self.split_data['X_train']), columns=comps,
                index=self.split_data['Yb_train'].index)
            df_train['Best'] = self.split_data['Yb_train']
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                pca.transform(self.split_data['X_test']), columns=comps,
                index=self.split_data['Yb_test'].index)
            df_test['Best'] = self.split_data['Yb_test']
            df_test['group'] = 'test'

            self.PlotProj['PCA'] = pd.concat([df_train,df_test],axis=0)

        else:
            pca.fit(self.X_s)
            comps = [f'Z{i+1}' for i in range(pca.n_components_)]

            df = pd.DataFrame(
                pca.transform(self.X_s), columns=comps, 
                index=self.performance.index)
            df['Best'] = self.performance['Best']
            self.PlotProj['PCA'] = df
        
        self.projections['PCA'] = pca

        if self.verbose:
            print("PCA projection added")

    def PILOT(self, mode, hd=False, split=True, num_params={}, yt=False, n_components=None, drop=False):
        """Fit PILOT and add projection to PlotProj. 
            Cannot do HD with numerical solver.

        Args:
            mode (str): 'a' for analytic solver, 'num' for numerical solver
            analytic (bool, optional): whether analytic or numerical solver used. Defaults to True.
            num_params (dict, optional): parameters for numerical solver. Dict keys - 'seed','ntries'.
        """

        lab = f'{mode}_y1' if yt else mode

        if f'PILOT{lab}' in self.projections.keys():
            print("PILOT projection already added")
            return
        if hd and mode == 'num':
            print('Cannot fit numeric HD projection')
            return
        
        if n_components==None:
            n_components = min(self.m, self.n) if hd else 2

        comps = [f'Z{i+1}' for i in range(n_components)]

        if mode=='a' and len(num_params) < 0:
            print('Warning: extra parameters given for analytic solver')
        if mode=='num' and len(num_params) == 0:
            print('Error: extra parameters required for numeric solver')
            return
        
        # allow parallel
        if self.parallel != False:
            num_params['parallel'] = self.parallel
        
        proj = Pilot()
        
        if split:
            if yt:
                proj.fit(self.split_data['X_train'], self.split_data['Yt_train'],
                     d=n_components, num_params=num_params)
            else:
                proj.fit(self.split_data['X_train'], self.split_data['Y_train'],
                     d=n_components, num_params=num_params)
                
            df_train = pd.DataFrame(
                proj.transform(self.split_data['X_train']), columns=comps, 
                index=self.split_data['Yb_train'].index)
            df_train['Best'] = self.split_data['Yb_train']
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                proj.transform(self.split_data['X_test']), columns=comps,
                index=self.split_data['Yb_test'].index)
            df_test['Best'] = self.split_data['Yb_test']
            df_test['group'] = 'test'
            
            self.PlotProj[f'PILOT{lab}'] = pd.concat([df_train,df_test],axis=0)

        else:
            if yt:
                proj.fit(self.X_s, self.Yt_s, d=n_components, num_params=num_params)
            else:
                proj.fit(self.X_s, self.Y_s, d=n_components, num_params=num_params)

            df = pd.DataFrame(
                proj.transform(self.X_s), columns=comps,
                index=self.performance.index)
            df['Best'] = self.performance['Best']
            self.PlotProj[f'PILOT{lab}'] = df
            
        self.projections[f'PILOT{lab}'] = proj

        if self.verbose:
            print(f"PILOT{lab} projection added")

    def PLS(self, mode='', hd=False, split=True, n_components=None, yt=False):

        lab = f'_y1' if (yt and mode == '') else mode

        if n_components==None:
            if hd:
                n_components = min(self.m, 
                    len(self.split_data['Yb_train']) if split else self.n)
            else:
                n_components = 2
        comps = [f'Z{i+1}' for i in range(n_components)]

        if f'PLSR{lab}' in self.projections.keys():
            print(f"PLSR{lab} projection already added")
            return
        if 'Best' not in self.performance.columns and mode == 'da':
            print("No classification data available for PLSDA projection")
            return        
        
        
        if mode == 'rel':
            if len(self.Y_rel) == 0:
                print("Relative performance data not available")
                return
            
            if split:
                Ytrain = self.split_data['Yr_train']
                Ytest = self.split_data['Yr_test']
            else:
                Y = self.Y_rel.values
        
        elif mode == 'da':
            if split:
                labs = self.performance['Best'].unique()
                Ytrain = pd.get_dummies(self.split_data['Yb_train'], 
                            columns=labs, dtype=int).values
                Ytest = pd.get_dummies(self.split_data['Yb_test'],
                            columns=labs, dtype=int).values
            else:
                Y = pd.get_dummies(self.performance['Best'], dtype=int).values

        elif mode =='' and yt:
            if split:
                Ytrain = self.split_data['Yt_train']
                Ytest = self.split_data['Yt_test']
            else:
                Y = self.Yt_s

        else:
            if split:
                Ytrain = self.split_data['Y_train']
                Ytest = self.split_data['Y_test']
            else:
                Y = self.Y_s

        pls = cross_decomposition.PLSRegression(n_components=n_components, 
                                scale=mode!='da')

        if split:
            pls.fit(self.split_data['X_train'], Ytrain)
            df_train = pd.DataFrame(
                pls.transform(self.split_data['X_train']), columns=comps,
                index=self.split_data['Yb_train'].index)
            df_train['Best'] = self.split_data['Yb_train']
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                pls.transform(self.split_data['X_test']), columns=comps,
                index=self.split_data['Yb_test'].index)
            df_test['Best'] = self.split_data['Yb_test']
            df_test['group'] = 'test'

            self.PlotProj[f'PLSR{lab}'] = pd.concat([df_train,df_test],axis=0)
        
        else:
            pls.fit(self.X_s, Y)
            df = pd.DataFrame(
                pls.transform(self.X_s), columns=comps,
                index=self.performance.index)
            df['Best'] = self.performance['Best']
            self.PlotProj[f'PLSR{lab}'] = df
            
        self.projections[f'PLSR{lab}'] = pls    

        if self.verbose:    
            print(f"PLSR{lab} projection added.")


    def LDA(self, hd=False, split=True, n_components=None, solver='svd'):

        n_components = self.performance['Best'].nunique()-1 if n_components == None else min(n_components, self.performance['Best'].nunique()-1)
        
        if hd or n_components > 2:
            print('HD proj not implemented')
            return
        if 'LDA' in self.projections.keys():
            print("LDA projection already added")
            return
        if 'Best' not in self.performance.columns:
            print("No classification data available for LDA projection")
            return
        
        lda = LinearDiscriminantAnalysis(solver=solver, n_components=n_components)

        if split:
            lda.fit(self.split_data['X_train'], self.split_data['Yb_train'])
            df_train = pd.DataFrame(
                lda.transform(self.split_data['X_train']), columns=[f'Z{i+1}' for i in range(n_components)],
                index=self.split_data['Yb_train'].index)
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                lda.transform(self.split_data['X_test']), columns=[f'Z{i+1}' for i in range(n_components)],
                index=self.split_data['Yb_test'].index)
            df_test['group'] = 'test'

            self.PlotProj['LDA'] = pd.concat([df_train,df_test],axis=0)

        else:
            lda.fit(self.X_s, self.performance['Best'])
            self.PlotProj['LDA'] = pd.DataFrame(
                lda.transform(self.X_s), columns=[f'Z{i+1}' for i in range(n_components)],
                index=self.performance.index)
            
        self.projections['LDA'] = lda        
        print(f"LDA projection added.")

    def AE(self, mode, hd=False,split=True, seed=111, n_components=2, yt=False, params={}):
        
        name = 'AElin' if mode == 'linear' else 'AEnon'
        name = name+'_y1' if yt else name
        if name in self.projections.keys():
            print(f"{name} projection already added")
            return
        
        if hd or n_components > 2:
            print('HD proj not implemented')
            return
        
        Y = self.split_data['Yt_train'] if (yt and split) else self.split_data['Y_train'] if split else self.Yt_s if yt else self.Y_s
        X = self.split_data['X_train'] if split else self.X_s
        
        if 'epochs' in params.keys():
            epochs = copy(params['epochs'])
            del(params['epochs'])
        else:
            epochs = 100

        ae = Projector(mode=mode, seed=seed, **params)
        proj = ae.fit(
                X,Y,validation_split=0, 
                epochs = epochs,
                dim=n_components
            )

        if split:
            
            df_train = pd.DataFrame(
                proj.instance_encoder.predict(X, verbose=0), 
                columns=['Z1','Z2'],
                index=self.split_data['Yb_train'].index)
            df_train['Best'] = self.split_data['Yb_train']
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                proj.instance_encoder.predict(self.split_data['X_test'], verbose=0), 
                columns=['Z1','Z2'],
                index=self.split_data['Yb_test'].index)
            df_test['Best'] = self.split_data['Yb_test']
            df_test['group'] = 'test'

            self.PlotProj[name] = pd.concat([df_train,df_test],axis=0)
        
        else:
            self.PlotProj[name] = pd.DataFrame(
                proj.instance_encoder.predict(X, verbose=0), 
                columns=['Z1','Z2'],
                index=self.performance.index)
            
        self.projections[name] = proj
        print(f"{name} projection added")
            
    def AEtune(self, hd=False,split=True, seed=111, tune_size=20, ntries=20, n_components=2, yt=False, params={}):
        '''only for nonlinear AE, tune parameters and fit'''

        name = 'AEnon' if not yt else 'AEnon_y1'
        if name in self.projections.keys():
            print(f"{name} projection already added")
            return
        
        if hd or n_components > 2:
            print('HD proj not implemented')
            return
        
        Y = self.split_data['Yt_train'] if (yt and split) else self.split_data['Y_train'] if split else self.Yt_s if yt else self.Y_s
        X = self.split_data['X_train'] if split else self.X_s
        
        if 'epochs' in params.keys():
            epochs = copy(params['epochs'])
            del(params['epochs'])
        else:
            epochs = 100 # just in case not in params but params is given

        ae_mod = Projector(mode='nonlinear', seed=seed, **params)
            
        if len(params) == 0:
            # this params is not the same as in Projector
            params = ae_mod.tune_nl(X, Y, max_trials=tune_size, epochs=100, validation_split=0.2, outpath=self.path)
            epochs = params['tuner/epochs'] if 'tuner/epochs' in params.keys() else 100        
                
        if self.verbose:
            print(f"Fitting AE with parameters: {params}")

        print(f'cores {self.parallel}')
        proj = ae_mod.multi_fit(
            X, Y,
            epochs = epochs,
            max_tries=ntries, validation_split=0,
            dim=n_components,
            n_cores= max(1,self.parallel))


        if split:
            
            df_train = pd.DataFrame(
                proj.instance_encoder.predict(X, verbose=0), 
                columns=['Z1','Z2'],
                index=self.split_data['Yb_train'].index)
            df_train['Best'] = self.split_data['Yb_train']
            df_train['group'] = 'train'

            df_test = pd.DataFrame(
                proj.instance_encoder.predict(self.split_data['X_test'], verbose=0), 
                columns=['Z1','Z2'],
                index=self.split_data['Yb_test'].index)
            df_test['Best'] = self.split_data['Yb_test']
            df_test['group'] = 'test'

            self.PlotProj[name] = pd.concat([df_train,df_test],axis=0)

        else:

            df = pd.DataFrame(
                proj.instance_encoder.predict(X, verbose=0), 
                columns=['Z1','Z2'],
                index=self.performance.index)
            df['Best'] = self.performance['Best']
            self.PlotProj[name] = df
            
        self.projections[name] = proj
        print(f"{name} projection added")

            
    def projectNewInstances(self, X_new, proj):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return
        
        # keep only features used in training
        X_new = X_new[self.featureNames]
        X_new_ind = X_new.index

        # scale
        if self.x_scaler is not None:
            X_new = self.x_scaler.transform(X_new.values)

        # project
        if 'AE' in proj:
            Z_new = pd.DataFrame(
                self.projections[proj].instance_encoder.predict(X_new, verbose=0),
                columns=['Z1','Z2'], index=X_new_ind)
        else:
            Z_new = pd.DataFrame(
                self.projections[proj].transform(X_new),
                columns=['Z1','Z2'], index=X_new_ind)
        
        return Z_new

    # delete projections
    def delProj(self, method):
        """Deletes projection from PlotProj and projections dict.

        Args:
            method (str): name of the projection to delete
        """
        if method in self.projections.keys():
            del self.projections[method]
            del self.PlotProj[method]
            print(f"{method} projection deleted")
        else:
            print(f"{method} projection not defined")    
    


def saveProjCoords(proj_data, path, extra_cols=[]):
    
    if isinstance(proj_data, dict): 
        proj_dict = {proj: proj_data[proj][['Z1','Z2']+extra_cols] 
                        for proj in proj_data.keys() if not proj.endswith('DA')}        
        pd.concat(
            proj_dict,names=['proj','instances']).reset_index().to_csv(path)
        
    # if proj_data is a DataFrame
    elif isinstance(proj_data, pd.DataFrame):
        proj_data.loc[(not proj_data['proj'].endswith('DA')),
                      ['Z1','Z2']+extra_cols].to_csv(path)


def all2D_projIS(metadata, best, obj_min, test_size, rand, ae_hps=None, cores=1):
    
    IS1 = InstanceSpace()
    IS1.fromMetadata(metadata,best=best)
    if cores > 1:
        IS1.doParallel(cores)

    IS1.getRelativePerf(obj_min)

    if type(test_size) == float:
        IS1.splitData(test_size,rand,scale=True,stratified=True)
    elif type(test_size) == dict:
        IS1.splitData_known(test_size['train'], test_size['test'], scale=True)
    else:
        print('test_size should be float or dict with train and test indices')
        return
    
    
    # Projections
    IS1.PCA()
    IS1.PILOT(mode='num', num_params={'seed':rand, 'ntries':5})
    IS1.PILOT(mode='num', num_params={'seed':rand, 'ntries':5}, yt=True)
    
    if ae_hps != None:
        for k,ae in ae_hps.items():
            IS1.AE(mode='nonlinear',yt=k.endswith('_y1'),params=ae, seed=rand) 


    IS1.PLS()
    IS1.PLS(mode='rel')
    
    
    return IS1


def get_coords_csv():
    path = './IS projections/'

    for d in os.listdir(path):
        if d.endswith('.pkl') and not os.path.exists(f'{path}coords/{d.removesuffix(".pkl")}.csv'):
            print(d)
            IS = pd.read_pickle(path+d)
            proj_df = pd.concat(
                [IS.PlotProj[p].rename(columns={'Z1': f'proj_{p}1', 'Z2': f'proj_{p}2'}
                ) for p in IS.PlotProj.keys()], axis=1
            )
            proj_df = proj_df.loc[:, ~proj_df.columns.duplicated('last')]
    
            pd.merge(proj_df, IS.performance,left_index=True, right_index=True
             ).to_csv(f'{path}coords/{d.removesuffix(".pkl")}.csv')


if __name__ == "__main__":
    
    pass

    # import time
    
    # with open('test-iSpace.pkl', 'rb') as f:
    #     iSpace0 = pkl.load(f)

    # ae_params = iSpace0.projections['AEnon'].hyperparameters

    # md = pd.read_csv('../tsp_all/metadata_train.csv', index_col=0)
    # iSpace = InstanceSpace()
    # iSpace.fromMetadata(md,scaler='s',best='Best',source='source')

    # iSpace.doParallel(30)
    # iSpace.doVerbose()

    # tic = time.time()
    # iSpace.PILOT(mode='num', split=False, num_params={'seed':111, 'ntries':30})
    # print('PILOT time:', time.time()-tic)

    # tic = time.time()
    # iSpace.AEtune(ntries=30, yt=False, split=False, tune_size=0, params=ae_params)
    # print('AE time:', time.time()-tic)

    # saveProjCoords(iSpace.PlotProj, 'test-proj_coords.csv',
    #                   extra_cols=['Best'])
    
    # with open('test-iSpace.pkl', 'wb') as f:
    #     pkl.dump(iSpace, f)

    