import pandas as pd
import numpy as np
import pickle as pkl
import os
from copy import deepcopy

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition, cross_decomposition
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from scipy.spatial import distance
from scipy import stats
from scipy.linalg import eig

from sklearn.model_selection import StratifiedKFold, GridSearchCV,RandomizedSearchCV, KFold, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.base import clone

# from predictions import fitmatsvm_quick, generate_params
from IS_class import InstanceSpace

import matplotlib.pyplot as plt
import seaborn as sns

import time
import multiprocessing as mp

pd.set_option('mode.chained_assignment',None)

class InstanceSpaceEval():

    def __init__(self, IS, split):
        # self.IS = IS
        self.proj_list = list(IS.projections.keys())
        self.evals = {}

        self.split = split

        if split:
            self.X_s = np.vstack([IS.split_data['X_train'], IS.split_data['X_test']])
            self.Y_s = np.vstack([IS.split_data['Y_train'], IS.split_data['Y_test']])
            self.X_split = {'train': IS.split_data['X_train'], 'test': IS.split_data['X_test']}
            self.Y_split = {'train': IS.split_data['Y_train'], 'test': IS.split_data['Y_test']}
            self.Yb_split = {'train': IS.split_data['Yb_train'], 'test': IS.split_data['Yb_test']}
            if len(IS.Yt_s) > 0:
                self.Yt_s = np.vstack([IS.split_data['Yt_train'], IS.split_data['Yt_test']])
                self.Yt_split = {'train': IS.split_data['Yt_train'], 'test': IS.split_data['Yt_test']}

            self.D_high_split = {
                'test_yt':distance.pdist(
                    np.hstack([self.X_split['test'], self.Yt_split['test']]), 
                    'euclidean'),
                'test_y': distance.pdist(
                    np.hstack([self.X_split['test'], self.Y_split['test']]), 
                    'euclidean'),
                'test': distance.pdist(
                    self.X_split['test'], 'euclidean'),
                'train_yt':distance.pdist(
                    np.hstack([self.X_split['train'], self.Yt_split['train']]), 
                    'euclidean'),
                'train_y': distance.pdist(
                    np.hstack([self.X_split['train'], self.Y_split['train']]), 
                    'euclidean'),
                'train': distance.pdist(
                    self.X_split['train'], 'euclidean')
            }
        else:
            self.X_s = IS.X_s
            self.Y_s = IS.Y_s
            self.Yb = IS.performance['Best']
            self.Yt_s = IS.Yt_s

        self.D_high = distance.pdist(self.X_s, 'euclidean')
        self.Dy_high = distance.pdist(np.hstack([self.X_s,self.Y_s]), 'euclidean')
        if len(IS.Yt_s) > 0:
            self.Dyt_high = distance.pdist(np.hstack([self.X_s,self.Yt_s]), 'euclidean')

        # add Best to PlotProj
        self.PlotProj = IS.PlotProj.copy()

        for proj in self.PlotProj.values():
            if 'Best' not in proj.columns:
                if 'group' in proj.columns:
                    proj['Best'] = pd.concat([
                        IS.split_data['Yb_train'], IS.split_data['Yb_test']])
                else:
                    proj['Best'] = IS.performance['Best']
            
            # drop na columns
            proj.dropna(axis=1, how='all', inplace=True)

    def intrinsic_dim_ratio(self):

        pca = decomposition.PCA()
        pca.fit(self.X_s)
        intrinsic_dim = np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1

        return intrinsic_dim / pca.n_components_
           
    def get_proj(self, proj):
        if proj not in self.proj_list:
            print(f"{proj} projection not available")
            return
        
        metrics = {
            m: self.evals[(p,m)] for (p,m) in self.evals.keys() if p == proj
        }

        if len(metrics) == 0:
            print(f"No evaluation metrics for {proj} projection")
            return
        
        return metrics
    
    def get_metric(self, metric):
        
        metrics = {
            p: self.evals[(p,m)] for (p,m) in self.evals.keys() if m == metric
        }

        if len(metrics) == 0:
            print(f"{metric} not calculated")
            return        
        
        return metrics

    def plot_metric(self, metric, legend=True):
        pass

    # projection evaluation
    def normalised_stress(self, proj, useY=False, dist=None):
        """
        Measures the preservation of point-pairwise distances from 
        HD to LD space.

        """

        projData = self.PlotProj[proj].copy()
        projData = projData[[c for c in projData.columns if c.startswith('Z')]]

        if dist == None:
        # distance matrices of original and projected data
            D_high = self.Dyt_high if (useY and proj.endswith('y1')) else self.Dy_high if useY else self.D_high
            D_low = distance.pdist(projData, 'euclidean')
        else:
            D_high, D_low = dist

        # normalised stress
        ns = np.sum((D_high - D_low)**2) / np.sum(D_high**2)
        
        lab = 'normalised_stress' if not useY else 'normalised_stressY'
        self.evals[(proj,lab)] = ns

    def normalised_stress_split(self, proj, test_mode, useY=False):
        """
        Measures the preservation of point-pairwise distances from 
        HD to LD space.

        """
        lab = 'normalised_stress' if not useY else 'normalised_stressY'

        projData = self.PlotProj[proj].copy()
        projData.reset_index(inplace=True)
        
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']

        # train
        D_high = self.D_high_split['train_yt'] if (useY and proj.endswith('y1')) else self.D_high_split['train_y'] if useY else self.D_high_split['train']
        
        D_low = distance.pdist(train[[c for c in train.columns if c.startswith('Z')]], 'euclidean')

        ns = np.sum((D_high - D_low)**2) / np.sum(D_high**2)
        self.evals[(proj,f'{lab}_train')] = ns

        if test_mode==0:
            return
        
        # test
        if test_mode==1:
        # measure on only test data
            D_high = self.D_high_split['test_yt'] if (useY and proj.endswith('y1')) else self.D_high_split['test_y'] if useY else self.D_high_split['test']
            D_low = distance.pdist(test[[c for c in test.columns if c.startswith('Z')]], 'euclidean')

        elif test_mode==2:
        # measure on test data, including train data as neighbors
        # same as unsplit method
            D_high = self.Dyt_high if (useY and proj.endswith('y1')) else self.Dy_high if useY else self.D_high
            D_low = distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')

        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
            print("Not a reasonable option for this metric")
            return
        
        ns = np.sum((D_high - D_low)**2) / np.sum(D_high**2)
        self.evals[(proj,f'{lab}_test')] = ns

        
    def neighbourhood_hit(self, proj, K, rel=False):

        # if 'Best' not in self.IS.performance.columns:
        #     print("No classification data available for evaluation")
        #     return        
        
        projData = self.PlotProj[proj].copy()
        y = projData['Best'].values

        # lab = 'NH_rel' if rel else 'NH'

        if K >= len(y):
            print("K exceeds number of instances")
            return     

        if not rel and (proj, 'NH') in self.evals.keys():
            print(f"NH already calculated for {proj} projection")
            return   

        D_low = distance.squareform(
            distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean'))
    
        nn_proj = D_low.argsort()
        knn_proj = nn_proj[:, 1:]

        hits_proj = {
            k: np.mean([np.mean(y[knn_proj[i,:k]] == y[i]) for i in range(len(y))]) for k in range(1,K+1)}

        self.evals[(proj, 'NH')] = hits_proj

        if rel:           

            D_high = distance.squareform(self.D_high)
            nn_orig = D_high.argsort()
            knn_orig = nn_orig[:, 1:]

            hits_orig = {
                k: np.mean([np.mean(y[knn_orig[i,:k]] == y[i]) for i in range(len(y))]) for k in range(1,K+1)}
            
            self.evals[(proj, 'NH_rel')] = {k: hits_proj[k] / hits_orig[k] for k in range(1,K+1)}    
       
    def neighbourhood_hit_split(self, proj, K, test_mode,rel=False):

        
        projData = self.PlotProj[proj].copy()
        y = projData['Best'].values
        projData.reset_index(inplace=True)
            
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']
        ytrain = train['Best'].values
        ytest = test['Best'].values

        if K >= len(ytest):
            print("K exceeds number of test instances")
            return 

        lab = 'NH_rel' if rel else 'NH'

        if not rel and (proj, f'{lab}_test{test_mode}') in self.evals.keys():
            print(f"{lab} already calculated for {proj} projection")
            return

        if (proj, f'{lab}_train') not in self.evals.keys():

            # train    
            D_low = distance.squareform(
                distance.pdist(train[[c for c in train.columns if c.startswith('Z')]], 'euclidean'))
        
            nn_proj = D_low.argsort()[:, 1:]
            hits_proj = {
                k: np.mean([np.mean(ytrain[nn_proj[i,:k]] == ytrain[i]) for i in range(len(ytrain))]) for k in range(1,K+1)
            }
            
            
            if rel:
                D_high = distance.squareform(self.D_high_split['train'])
                nn_orig = D_high.argsort()[:, 1:]

                hits_orig = {
                    k: np.mean([np.mean(ytrain[nn_orig[i,:k]] == ytrain[i]) for i in range(len(ytrain))]) for k in range(1,K+1)
                }
                self.evals[(proj,f'NH_rel_train')] = {k: hits_proj[k] / hits_orig[k] for k in range(1,K+1)}
        
            self.evals[(proj,f'NH_train')] = hits_proj

        if test_mode==0:
            return
        
        # test
        if test_mode==1:
        # measure on only test data
            D_low = distance.squareform(
                distance.pdist(test[[c for c in test.columns if c.startswith('Z')]], 'euclidean'))
        
            nn_proj = D_low.argsort()[:, 1:]            
            hits_proj = {
                k: np.mean([np.mean(ytest[nn_proj[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
            }            
            
        elif test_mode==2:
        # measure on test data, including train data as neighbors
            D_low = distance.squareform(
                distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')
                )[test.index]
        
            nn_proj = D_low.argsort()[:, 1:]            
            hits_proj = {
                k: np.mean([np.mean(y[nn_proj[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
            }
            
        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
            D_low = distance.squareform(
                distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')
                )[test.index]
        
            nn_proj = D_low.argsort()
            knn_proj = np.array([[i for i in r if i in train.index] for r in nn_proj])[:, 1:]

            hits_proj = {
                k: np.mean([np.mean(y[knn_proj[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
            }
            
        self.evals[(proj,f'NH_test{test_mode}')] = hits_proj
        

        if rel:

            if test_mode==1:
                D_high = distance.squareform(self.D_high_split['test'])
            
                nn_orig = D_high.argsort()[:, 1:]
                hits_orig = {
                    k: np.mean([np.mean(ytest[nn_orig[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
                }
                
            elif test_mode==2:
                D_high = distance.squareform(self.D_high)[test.index]

                nn_orig = D_high.argsort()[:, 1:]
                hits_orig = {
                    k: np.mean([np.mean(y[nn_orig[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
                }                
                
            elif test_mode==3:
                D_high = distance.squareform(self.D_high)[test.index]
                
                nn_orig = D_high.argsort()
                knn_orig = np.array(
                    [[i for i in r if i in train.index] for r in nn_orig])[:, 1:]

                hits_orig = {
                    k: np.mean([np.mean(y[knn_orig[i,:k]] == ytest[i]) for i in range(len(ytest))]) for k in range(1,K+1)
                }

            self.evals[(proj,f'NH_rel_test{test_mode}')] = {k: hits_proj[k] / hits_orig[k] for k in range(1,K+1)}
            
            
    def corr_ratio(self, proj, dist=None):
        # do like shepard goodness but do D-high vs D-alg
        # and D-low vs D-alg
        # ratio of the two

        projData = self.PlotProj[proj].copy()
        projData = projData[[c for c in projData.columns if c.startswith('Z')]]

        if dist == None:
            D_high = self.D_high
            D_low = distance.pdist(projData, 'euclidean')
        else:
            D_high, D_low = dist
        
        D_alg = distance.pdist(
            self.Yt_s if proj.endswith('y1') else self.Y_s, 
            'euclidean')

        # goodness
        good_hd = stats.spearmanr(D_high, D_alg, axis=None)[0]
        good_ld = stats.spearmanr(D_low, D_alg, axis=None)[0]

        self.evals[(proj,'corr_ratio')] = good_ld / good_hd
        # {'good_hd': good_hd, 'good_ld': good_ld}

    def corr_ratio_split(self, proj, test_mode):

        projData = self.PlotProj[proj].copy()
        projData.reset_index(inplace=True)
        
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']

        # train
        D_high = self.D_high_split['train']
        D_low = distance.pdist(train[[c for c in train.columns if c.startswith('Z')]], 'euclidean')
        
        D_alg = distance.pdist(
            self.Yt_split['train'] if proj.endswith('y1') else self.Y_split['train'], 
            'euclidean') 
        
        good_hd = stats.spearmanr(D_high, D_alg, axis=None)[0]
        good_ld = stats.spearmanr(D_low, D_alg, axis=None)[0]
        self.evals[(proj,'corr_ratio_train')] = good_ld / good_hd

        if test_mode==0:
            return
        
        # test
        if test_mode==1:
        # measure on only test data
            D_high = self.D_high_split['test']
            D_low = distance.pdist(test[[c for c in test.columns if c.startswith('Z')]], 'euclidean')
        
            D_alg = distance.pdist(
                self.Yt_split['test'] if proj.endswith('y1') else self.Y_split['test'], 
                'euclidean')
        
        elif test_mode==2:
        # measure on test data, including train data as neighbors
        # same as unsplit method
            D_high = self.D_high
            D_low = distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')

            D_alg = distance.pdist(self.Yt_s if proj.endswith('y1') else self.Y_s, 'euclidean')

        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
            print("Not a reasonable option for this metric")
            return
        
        good_hd = stats.spearmanr(D_high, D_alg, axis=None)[0]
        good_ld = stats.spearmanr(D_low, D_alg, axis=None)[0]
        self.evals[(proj,'corr_ratio_test')] = good_ld / good_hd
        

    def shepard_goodness(self, proj, useY=False, dist=None):
        """
        Shepard diagram is a scatterplot of the pairwise distances in the the HD vs LD.
        The goodness of fit is measured by the Spearman correlation.
        """

        projData = self.PlotProj[proj].copy()
        projData = projData[[c for c in projData.columns if c.startswith('Z')]]

        if dist == None:
        # distance matrices of original and projected data
            D_high = self.Dy_high if useY else self.D_high
            D_low = distance.pdist(projData, 'euclidean')
        else:
            D_high, D_low = dist
            
        # goodness
        sg = stats.spearmanr(D_high, D_low, axis=None)[0]

        lab = 'shepard_goodness' if not useY else 'shepard_goodnessY'
        self.evals[(proj,lab)] = sg

    def shepard_goodness_split(self, proj, test_mode, useY=False):
        """
        Shepard diagram is a scatterplot of the pairwise distances in the the HD vs LD.
        The goodness of fit is measured by the Spearman correlation
        """

        lab = 'shepard_goodness' if not useY else 'shepard_goodnessY'

        projData = self.PlotProj[proj].copy()
        projData.reset_index(inplace=True)
            
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']
        

        # train
        D_high = self.D_high_split['train_yt'] if (useY and proj.endswith('y1')) else self.D_high_split['train_y'] if useY else self.D_high_split['train']
        D_low = distance.pdist(train[[c for c in train.columns if c.startswith('Z')]], 'euclidean')

        sg = stats.spearmanr(D_high, D_low, axis=None)[0]
        self.evals[(proj,f'{lab}_train')] = sg

        if test_mode==0:
            return
        
        # test
        if test_mode==1:
        # measure on only test data
            D_high = self.D_high_split['test_yt'] if (useY and proj.endswith('y1')) else self.D_high_split['test_y'] if useY else self.D_high_split['test']
            D_low = distance.pdist(test[[c for c in test.columns if c.startswith('Z')]], 'euclidean')

        elif test_mode==2:
        # measure on test data, including train data as neighbors
        # same as unsplit method
            D_high = self.Dyt_high if (useY and proj.endswith('y1')) else self.Dy_high if useY else self.D_high
            D_low = distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')

        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
            print("Not a reasonable option for this metric")
            return
        
        sg = stats.spearmanr(D_high, D_low, axis=None)[0]
        self.evals[(proj,f'{lab}_test')] = sg
        
    
    def metrics_k0(self,k, orig, proj):
        print(f'k {k}', end='\r')
        n = len(orig)
        knn_orig = orig[:, 1:k + 1]
        knn_proj = proj[:, 1:k + 1]

        U = [np.setdiff1d(knn_proj[i], knn_orig[i],True) for i in range(n)]
        # U = [set(knn_proj[i]).difference(set(knn_orig[i])) for i in range(n)]
        u_ranks = [sum([(orig[i].tolist().index(u) - k) for u in U[i]]) for i in range(n)]
        sum_T = sum(u_ranks)

        V = [np.setdiff1d(knn_orig[i], knn_proj[i],True) for i in range(n)]
        # V = [set(knn_orig[i]).difference(set(knn_proj[i])) for i in range(n)]
        v_ranks = [sum([(proj[i].tolist().index(v) - k) for v in V[i]]) for i in range(n)]
        sum_C = sum(v_ranks)            
        
        mult = 2 / (n*k * (2*n - 3*k - 1))

        return {'trustworthiness':1 - mult * sum_T, 
                'continuity':1 - mult * sum_C}
    
    def metrics_k(self,k, orig, proj):
        print(f'k {k}', end='\r')
        n = len(orig)
        knn_orig = orig[:, 1:k + 1]
        knn_proj = proj[:, 1:k + 1]

        sum_it = 0
        sum_ic = 0

        for i in range(n):
            U = np.setdiff1d(knn_proj[i], knn_orig[i])

            sum_jt = 0
            for j in range(U.shape[0]):
                sum_jt += np.where(orig[i] == U[j])[0] - k

            sum_it += sum_jt

            V = np.setdiff1d(knn_orig[i], knn_proj[i])

            sum_jc = 0
            for j in range(V.shape[0]):
                sum_jc += np.where(proj[i] == V[j])[0] - k

            sum_ic += sum_jc

        mult = 2 / (n * k * (2 * n - 3 * k - 1))
        return {'trustworthiness':float((1 - (mult * sum_it)).squeeze()),
                'continuity':float((1 - (mult * sum_ic)).squeeze())}



    def trust_cont(self, proj, K):
        
        projData = self.PlotProj[proj].copy()
        projData = projData[[c for c in projData.columns if c.startswith('Z')]]
        
        D_low = distance.squareform(distance.pdist(projData, 'euclidean'))
        D_high = distance.squareform(self.D_high)
            
        nn_orig = D_high.argsort()
        nn_proj = D_low.argsort()

        trust_cont = {k: self.metrics_k(k,nn_orig,nn_proj) for k in range(1,K+1)}
        self.evals[(proj,'trust')] = {
            k: trust_cont[k]['trustworthiness'] for k in range(1,K+1)
        }
        self.evals[(proj,'continuity')] = {
            k: trust_cont[k]['continuity'] for k in range(1,K+1)
        }
         
    def trust_cont_split(self, proj, K, test_mode):        
        
        projData = self.PlotProj[proj].copy()
        projData.reset_index(inplace=True)
        
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']

        if K >= len(test):
            print("K exceeds number of test instances")
            return         
        
        # train
        D_low = distance.squareform(
            distance.pdist(train[[c for c in train.columns if c.startswith('Z')]], 'euclidean'))
        D_high = distance.squareform(self.D_high_split['train'])
        
        nn_orig = D_high.argsort()
        nn_proj = D_low.argsort()

        trust_cont = {k: self.metrics_k(k,nn_orig,nn_proj) for k in range(1,K+1)}
        self.evals[(proj,f'trust_train')] = {
            k: trust_cont[k]['trustworthiness'] for k in range(1,K+1)
        }
        # print('', end='\r')
        self.evals[(proj,f'continuity_train')] = {
            k: trust_cont[k]['continuity'] for k in range(1,K+1)
        }
        # print('', end='\r')

        if test_mode==0:
            return

        # test
        if test_mode==1:
        # measure on only test data
            D_low = distance.squareform(
                distance.pdist(test[[c for c in test.columns if c.startswith('Z')]], 'euclidean'))
            D_high = distance.squareform(self.D_high_split['test'])
        
            nn_orig = D_high.argsort()
            nn_proj = D_low.argsort()
        
        elif test_mode==2:
        # measure on test data, including train data as neighbors
            D_low = distance.squareform(
                distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')
                )[test.index]
            D_high = distance.squareform(self.D_high)[test.index]

            nn_orig = D_high.argsort()
            nn_proj = D_low.argsort()

        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
        #     D_low = distance.squareform(
        #         distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')
        #         )[test.index]
        #     D_high = distance.squareform(self.D_high)[test.index]

        #     nn_orig = D_high.argsort()
        #     nn_proj = D_low.argsort()
        # not implemented - would need to change metrics_k calc for knn_proj
            print("Option not implemented for this metric")
            return

        trust_cont = {k: self.metrics_k(k,nn_orig,nn_proj) for k in range(1,K+1)}
        self.evals[(proj,f'trust_test{test_mode}')] = {
            k: trust_cont[k]['trustworthiness'] for k in range(1,K+1)
        }
        # print('', end='\r')
        self.evals[(proj,f'continuity_test{test_mode}')] = {
            k: trust_cont[k]['continuity'] for k in range(1,K+1)
        }
        # print('', end='\r')
        

    def uniformity(self, proj, rel=False):

        # if 'Best' not in self.IS.performance.columns:
        #     print("No classification data available for evaluation")
        #     return        
        
        projData = self.PlotProj[proj].copy()
        projData = projData[[c for c in projData.columns if c.startswith('Z')]]

        if not rel and (proj, 'uniformity') in self.evals.keys():
            print(f"Uniformity already calculated for {proj} projection")
            return   

        D_low = distance.squareform(
            distance.pdist(projData, 'euclidean'))
        nn_Dl = np.sort(D_low)[:,1]
        
        # self.evals[(proj, 'uniformity')] = 1 - np.std(nn_Dl)/np.mean(nn_Dl)
        self.evals[(proj, 'uniformity')] = np.std(nn_Dl)/np.mean(nn_Dl)


        if rel:           

            D_high = distance.squareform(self.D_high)
            nn_Dh = np.sort(D_high)[:,1]

            # self.evals[(proj, 'uniformity_rel')] = (1 - np.std(nn_Dl)/np.mean(nn_Dl)) / (1 - np.std(nn_Dh)/np.mean(nn_Dh))  
            self.evals[(proj, 'uniformity_rel')] = (np.std(nn_Dl)/np.mean(nn_Dl)) / (np.std(nn_Dh)/np.mean(nn_Dh))  
       
    def uniformity_split(self, proj, test_mode,rel=False):

        
        projData = self.PlotProj[proj].copy()
        projData.reset_index(inplace=True)
        
        train = projData[projData['group'] == 'train']
        test = projData[projData['group'] == 'test']

        
        lab = 'uniformity_rel' if rel else 'uniformity'

        if not rel and (proj, f'{lab}_test{test_mode}') in self.evals.keys():
            print(f"{lab} already calculated for {proj} projection")
            return

        if (proj, f'{lab}_train') not in self.evals.keys():

            # train    
            D_low = distance.squareform(distance.pdist(
                    train[[c for c in train.columns if c.startswith('Z')]], 'euclidean'))
            nn_Dl = np.sort(D_low)[:,1]

            # self.evals[(proj,f'uniformity_train')] = 1 - np.std(nn_Dl)/np.mean(nn_Dl)
            self.evals[(proj,f'uniformity_train')] = np.std(nn_Dl)/np.mean(nn_Dl)
            
            if rel:
                D_high = distance.squareform(self.D_high_split['train'])
                nn_Dh = np.sort(D_high)[:,1]

                # self.evals[(proj,f'uniformity_rel_train')] = (1 - np.std(nn_Dl)/np.mean(nn_Dl)) / (1 - np.std(nn_Dh)/np.mean(nn_Dh))
                self.evals[(proj,f'uniformity_rel_train')] = (np.std(nn_Dl)/np.mean(nn_Dl)) / (np.std(nn_Dh)/np.mean(nn_Dh))

        if test_mode==0:
            return
        
        # test
        if test_mode==1:
        # measure on only test data
            D_low = distance.squareform(distance.pdist(
                test[[c for c in test.columns if c.startswith('Z')]], 'euclidean'))
             
        elif test_mode==2:
        # measure on test data, including train data as neighbors
            D_low = distance.squareform(
                distance.pdist(projData[[c for c in projData.columns if c.startswith('Z')]], 'euclidean')
                )[test.index]
            
        elif test_mode==3:
        # measure on test data - including only 1 test instance at a time
            print("Not a reasonable option for this metric")
            return
            
        nn_Dl = np.sort(D_low)[:,1]
        # self.evals[(proj,f'uniformity_test{test_mode}')] = 1 - np.std(nn_Dl)/np.mean(nn_Dl)
        self.evals[(proj,f'uniformity_test{test_mode}')] = np.std(nn_Dl)/np.mean(nn_Dl)

        if rel:

            if test_mode==1:
                D_high = distance.squareform(self.D_high_split['test'])
                            
            elif test_mode==2:
                D_high = distance.squareform(self.D_high)[test.index]

            elif test_mode==3:
                print("Not a reasonable option for this metric")
                return
            
            nn_Dh = np.sort(D_high)[:,1]
            # self.evals[(proj,f'uniformity_rel_test{test_mode}')] = (1 - np.std(nn_Dl)/np.mean(nn_Dl)) / (1 - np.std(nn_Dh)/np.mean(nn_Dh))
            self.evals[(proj,f'uniformity_rel_test{test_mode}')] = (np.std(nn_Dl)/np.mean(nn_Dl)) / (np.std(nn_Dh)/np.mean(nn_Dh))

    # TODO - figure a good w. should be a matrix
    def normalised_stress_weighted(self, proj, w, dist=None):        

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist

        # normalised stress 
        # deliberately not doing matmult - we just want termwise mult 
        return np.sum(w * (D_high - D_low)**2) / np.sum(w * D_high**2)
    
    def proj_evaluation(self, proj='all'):

        if proj == 'all':
            proj_list = self.proj_list
        elif isinstance(proj, list):
            proj_list = [p for p in proj if p in self.proj_list]
            if len(proj_list) == 0:
                print("None of the specified projections are available")
                return
        elif isinstance(proj, str):
            proj_list = [proj] if proj in self.proj_list else []
        else:
            print("proj should be a string, list of strings, or 'all'")
            return

        for proj in proj_list:
            # self.normalised_stress_split(proj, 2, useY=False)
            # self.normalised_stress_split(proj, 2, useY=True)
            self.shepard_goodness(proj, useY=False)
            self.shepard_goodness(proj, useY=True)
            self.shepard_goodness_split(proj, 2, useY=False)
            self.shepard_goodness_split(proj, 2, useY=True)
            self.corr_ratio(proj)
            self.corr_ratio_split(proj, 2)
            self.uniformity(proj, rel=True)
            self.uniformity_split(proj, 2, rel=True)
            K = int(np.round(len(self.X_split['train'])*0.05))
            # self.trust_cont_split(proj, K, 0)
            # self.trust_cont(proj, K)
            self.neighbourhood_hit(proj, K, rel=True)
            self.neighbourhood_hit_split(proj, K, 2, rel=True)
    

class PredictionEval():

    def __init__(self, proj_data, split):
        
        if isinstance(proj_data, dict): 
            self.projections = deepcopy(proj_data)

        elif isinstance(proj_data, pd.DataFrame):
            projCols = [c for c in proj_data.columns if c != 'proj']
            self.projections = {
                proj: proj_data.loc[proj_data['proj'] == proj, projCols] for proj in proj_data['proj'].unique()
            }

        self.split = split
        self.evals = {}
        self.models = {}
        self.regrets = {}


    ### predictions from DA - no CV

    ## adding prediction - SVM
    def makePredictions_svm(self, proj, params):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return

        svm = SVC(**params, probability=True)
        Z = self.projections[proj]
        cols = [c for c in Z.columns if c.startswith('Z')]
        if len(cols) == 0:
            print(f"No projection columns found for {proj}")
            return
        
               
        if self.split:
            svm.fit(Z.loc[Z['group']=='train',cols].values, 
                    Z.loc[Z['group']=='train','Best'])
            
        else:
            svm.fit(Z[cols].values, Z['Best'])    
        
        self.projections[proj]['pred_svm'] = svm.predict(Z[cols].values)
        self.projections[proj][['prob_svm_'+c for c in svm.classes_]] = svm.predict_proba(Z[cols].values)
        self.models[(proj,'svm')] = svm

    def makePredictions_knn(self, proj, params):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return

        knn = KNeighborsClassifier(**params)
        Z = self.projections[proj]
        cols = [c for c in Z.columns if c.startswith('Z')]
        if len(cols) == 0:
            print(f"No projection columns found for {proj}")
            return
        
        if self.split:
            knn.fit(Z.loc[Z['group']=='train',cols].values, 
                    Z.loc[Z['group']=='train','Best'])            
           
        else:
            knn.fit(Z[cols].values, Z['Best'])    
        
        self.projections[proj]['pred_knn'] = knn.predict(Z[cols].values)
        self.projections[proj][['prob_knn_'+c for c in knn.classes_]] = knn.predict_proba(Z[cols].values)

        self.models[(proj,'knn')] = knn

    def makePredictions_log(self, proj, cv, penalty='l1', C=1, max_iter=10000):
        
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return
        
        X = self.projections[proj]
        cols = [c for c in X.columns if c.startswith('Z')]
        if len(cols) == 0:
            print(f"No projection columns found for {proj}")
            return
        
        mod_name = 'log' if penalty==None else f'log+{penalty}'

        if cv:
            lr = LogisticRegressionCV(penalty=penalty, solver='saga', 
                    Cs=np.logspace(-4,4,9),max_iter=max_iter, cv=5, random_state=1111)
        else:
            lr = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, solver='saga', random_state=1111)

        if self.split:
            lr.fit(X.loc[X['group']=='train',cols].values, 
                   X.loc[X['group']=='train','Best'])
        else:
            lr.fit(X[cols].values, X['Best'])

        self.projections[proj][f'pred_{mod_name}'] = lr.predict(X[cols].values)
        self.models[(proj,mod_name)] = lr
           

    def makePredictions_avg(self, avg_algo):
        if 'All' not in self.projections.keys():
            print(f"All projection not available")
            return
        self.projections['All']['pred_avg'] = np.repeat(avg_algo, len(self.projections['All']))

    def makePredictions_HD(self, proj, avg_algo,train, test, params):
        svm = SVC(**params['svm'], probability=True)
        knn = KNeighborsClassifier(**params['knn'])

        svm.fit(train['X'], train['Yb'])
        knn.fit(train['X'], train['Yb'])

        predDF = {
            'Best': np.concatenate([train['Yb'], test['Yb']]),
            'group': np.concatenate([np.repeat('train',len(train['Yb'])), np.repeat('test',len(test['Yb']))]),
            'pred_svm': svm.predict(np.vstack([train['X'], test['X']])),
            'pred_knn': knn.predict(np.vstack([train['X'], test['X']])),
            'pred_avg': np.repeat(avg_algo, len(train['Yb'])+len(test['Yb']))
        }
        predDF = pd.DataFrame(predDF,
                        index=list(train['Yb'].index) + list(test['Yb'].index))
        
        predDF[['prob_svm_'+c for c in svm.classes_]] = svm.predict_proba(np.vstack([train['X'], test['X']]))
        predDF[['prob_knn_'+c for c in knn.classes_]] = knn.predict_proba(np.vstack([train['X'], test['X']]))

        predDF.index.name = 'instances'

        self.projections[proj] = predDF

    
    def evaluate_predictions0(self, proj):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return
        
        Z = self.projections[proj]
        mod_list = [m.removeprefix('pred_') for m in Z.columns if m.startswith('pred')]

        for t in ['train','test']:
            act = Z.loc[Z['group']==t,'Best']

            for mod in mod_list:        
                pred = Z.loc[Z['group']==t,'pred_'+mod]
                probs = Z.loc[Z['group']==t,
                              [c for c in Z.columns if c.startswith('prob_'+mod)]].values

                self.evals[(proj,mod,t)] = {
                    'accuracy': accuracy_score(act, pred),
                    'precision': precision_score(act, pred,zero_division=1,average='weighted'),
                    'recall': recall_score(act, pred,zero_division=1,average='macro'),
                    'auc': roc_auc_score(act, probs[:,1]) if probs.shape[1] == 2 else 
                            np.nan if probs.shape[1] == 0 else
                            roc_auc_score(act,probs, multi_class='ovr')
                    }
                
    def evaluate_predictions(self, proj):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return
        
        Z = self.projections[proj]
        mod_list = [m.removeprefix('pred_') for m in Z.columns if m.startswith('pred')]

        for t in ['train','test']:
            act = Z.loc[Z['group']==t,'Best']
            classes = np.unique(act)

            for mod in mod_list:        
                pred = Z.loc[Z['group']==t,'pred_'+mod]
                # probs = Z.loc[Z['group']==t,
                #               [c for c in Z.columns if c.startswith('prob_'+mod)]].values

                self.evals[(proj,mod,t)] = {
                    'accuracy': accuracy_score(act, pred),                    
                    'precision': precision_score(act, pred, zero_division=1,pos_label=classes[0] if len(classes)==2 else 1,
                                                 average='binary' if len(classes)==2 else 'macro'),
                    'precision_w': precision_score(act, pred, zero_division=1,average='weighted'),
                    'recall': recall_score(act, pred, zero_division=1,pos_label=classes[0] if len(classes)==2 else 1,
                                           average='binary' if len(classes)==2 else 'macro'),
                    'recall_m': recall_score(act, pred, zero_division=1,average='macro'),
                    'f1': f1_score(act, pred, zero_division=1,average='macro'),
                    'f1_w': f1_score(act, pred, zero_division=1,average='weighted'),
                    # 'conf': confusion_matrix(act, pred, labels=classes),
                    # 'auc': roc_auc_score(act, probs[:,1]) if probs.shape[1] == 2 else 
                    #         np.nan if probs.shape[1] == 0 else
                    #         roc_auc_score(act,probs, multi_class='ovr')
                    }

    def calc_regrets(self, proj, perf_data, min, tie_lab = None):

        perf_data = perf_data[[c for c in perf_data.columns if c.startswith('algo_')]].copy()

        if tie_lab != None:
            # perf for tied column is worst case (if actually tied it doesn't matter)
            perf_data['algo_'+tie_lab] = perf_data.max(axis=1) if min else perf_data.min(axis=1)
        
        if min:
            Ydiff = perf_data.apply(
                lambda row: row - row.min(), axis=1
            ).fillna(0)#.values
            Yrel = perf_data.apply(
                lambda row: row if row.min()==0 else np.abs((row - row.min())/row.min()), axis=1
            ).fillna(0)#.values

        else:
            Ydiff = perf_data.apply(
                lambda row: row - row.max(), axis=1
            ).fillna(0)#.values
            Yrel = perf_data.apply(
                lambda row: row if row.max()==0 else np.abs((row - row.max())/row.max()), axis=1
            ).fillna(0)

        Yrel = Yrel.join(Ydiff, how='left', rsuffix='_abs')
        
        
        for best in [b for b in self.projections[proj].columns if b.startswith('pred_')]:
                
            Yr_proj = Yrel.copy()
            Yr_proj[['bestA','bestP','group']] = self.projections[proj][['Best', best,'group']]
                
            ## check how many misclassifications there are
                
            Yr_proj['reg'] = Yr_proj.apply(lambda x: x[f'algo_{x["bestP"]}'], axis=1)
            Yr_proj['reg_sub'] = Yr_proj.apply(lambda x: x['reg'] if x['bestA']!=x['bestP'] else np.nan, axis=1)
                
            Yr_proj['abs_reg'] = Yr_proj.apply(lambda x: x[f'algo_{x["bestP"]}_abs'], axis=1)
            Yr_proj['abs_reg_sub'] = Yr_proj.apply(lambda x: x['abs_reg'] if x['bestA']!=x['bestP'] else np.nan, axis=1)
                        
            Yr_proj.drop(columns=[c for c in Yr_proj.columns if c.startswith('algo_')], inplace=True)
            self.regrets[(proj,best.removeprefix('pred_'))] = Yr_proj
        


def _evaluate_params(estimator, iSpace, proj, idx, params, scoring, scale):
    """
    Evaluates a single fold for a given parameter set.
    """
    train_idx, test_idx = idx

    if proj == 'All':
        X = np.vstack([iSpace.split_data['X_train'],iSpace.split_data['X_test']])
        y = np.concatenate([iSpace.split_data['Yb_train'],iSpace.split_data['Yb_test']])
    else:
        X = iSpace.PlotProj[proj][['Z1','Z2']].values
        y = iSpace.PlotProj[proj]['Best'].values      

    # scale X
    if scale:
        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        X = scaler.transform(X)       

    model = clone(estimator)
    model.set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    score = scoring(y[test_idx], preds)

    return (proj, params, score)   

def _evaluate_params_basic(estimator, iSpace, proj, idx, params, scoring, scale):
    """
    Evaluates a single fold for a given parameter set.
    """
    train_idx, test_idx = idx
    #print(proj,end='\r')

    if proj == 'All':
        X = iSpace.split_data['X_train']
        y = iSpace.split_data['Yb_train'].values
    else:
        X = iSpace.PlotProj[proj].query('group=="train"')[['Z1','Z2']].values
        y = iSpace.PlotProj[proj].query('group=="train"')['Best'].values      

    # scale X
    if scale:
        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        X = scaler.transform(X)       

    model = clone(estimator)
    model.set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    score = scoring(y[test_idx], preds)

    return (proj, params, score)   

def grid_search_cv(estimator, iSpace, param_grid, indices, n_cores=1, scoring=accuracy_score, scale=False, basic=False):
    """
    Custom grid search with multiprocessing and external loop over folds.

    Parameters:
    - estimator: Scikit-learn estimator to evaluate.
    - iSpace: InstanceSpace object containing the data.
    - param_grid: List of dicts with parameters to evaluate.
    - indices: Tuple with train-test indices.
    - n_cores: Number of cores for parallel processing.
    - scoring: Scoring function to evaluate predictions.

    Returns:
    - DataFrame with scores for each parameter set and projection.
    """

    proj_keys = list(iSpace.PlotProj.keys())+['All']

    if basic:
        _evaluate_params_func = _evaluate_params_basic
    else:
        _evaluate_params_func = _evaluate_params

    # Run all tasks
    if n_cores > 1:
        pool = mp.Pool(n_cores)

        results = [pool.apply_async(_evaluate_params_func,
                    args=(estimator, iSpace, proj, indices, params, scoring, scale)) 
                    for proj in proj_keys for params in param_grid]
        pool.close()
        pool.join()

        results = [r.get() for r in results]

    else:
        results = [_evaluate_params_func(estimator, iSpace, proj, indices, params, scoring, scale)
                    for proj in proj_keys for params in param_grid]
    
    # Convert to DataFrame for aggregation
    score_df = pd.DataFrame([
        {**params, 'proj':proj, 'score': score}
        for proj, params, score in results
    ])
    
    return score_df

def pred_cv(metadata, selFeats, outpath, iSpace=None):
    '''metadata should be training data only'''

    train_ind = list(metadata.index)
    other_cols = [c for c in metadata.columns if not c.startswith('feature_')]

    cores = max(iSpace.parallel,1) if iSpace is not None else 1
    
    def generate_params_(min, max):  

        rbf = [{'C': 10**i, 'gamma': 10**j, 'kernel': 'rbf'} 
                    for i in range(min, max+1) for j in range(min, max+1)]
        
        poly = [{'C': 10**i, 'degree': j, 'kernel': 'poly'} 
                    for i in range(min, max+1) for j in range(1,4)]
        
        return  rbf + poly    
            
    
    #param_space = generate_params_LH(30,1111)
    param_space = generate_params_(-3,3)

    param_keys = [p.keys() for p in param_space]
    param_keys = list(set().union(*param_keys))  # Unique keys across all param sets

    mK = round(len(train_ind)*0.05)

    if iSpace is not None:
        ae_hps = {
            k: v.hyperparameters for k,v in iSpace.projections.items() if k.startswith('AEnon')}

    cv_scores_svm = []
    cv_scores_knn = []

    gen_seed = np.random.default_rng(1111)

    # repeated cross-validation

    for f in range(2):
        print(f'Repeat {f}')
        
        cv = [inds for inds in StratifiedKFold(n_splits=5, shuffle=True, random_state=gen_seed.integers(1,100)).split(
            metadata[selFeats], metadata['Best']
        )]

        for i in range(5):
            tic = time.time()
            print(f'Fold {i}')
            cv_train = [train_ind[j] for j in cv[i][0]]
            cv_test = [train_ind[j] for j in cv[i][1]]
            tt = (np.arange(len(cv_train)), np.arange(len(cv_train),len(cv_train)+len(cv_test)))

            IS_cv = InstanceSpace()
            IS_cv.fromMetadata(metadata[selFeats+other_cols], 
                            scaler='s',best='Best')
            IS_cv.splitData_known(cv_train, cv_test, True)

            # all projections
            IS_cv.PCA()
            IS_cv.PILOT(mode='num', num_params={'seed':111, 'ntries':1})
            IS_cv.PILOT(mode='num', num_params={'seed':111, 'ntries':1}, yt=True)        
            IS_cv.PLS()
            # IS_cv.PLS(yt=True)
            IS_cv.getRelativePerf(min=True)
            IS_cv.PLS(mode='rel')

            # AE projections
            if iSpace is not None:
                for k,ae in ae_hps.items():
                    IS_cv.AE(mode='nonlinear',yt=k.endswith('_y1'),params=ae) 

            toc = time.time()
            print(f'Projections time: {toc-tic}')
            svm_t = 0
            knn_t = 0

            #### SVM CV
            tic = time.time()
            svm_scores = grid_search_cv(
                estimator=SVC(random_state=1111,probability=False,coef0=1),
                iSpace=IS_cv, param_grid=param_space, indices=tt, 
                n_cores=cores, scale=True
            )
            toc = time.time()
            print(f'SVM time: {toc-tic}', end='\r')
            svm_t += toc-tic

            #### knn CV
            tic = time.time()
            knn_scores = grid_search_cv(
                estimator=KNeighborsClassifier(),
                iSpace=IS_cv,
                param_grid=[{'n_neighbors': p} for p in np.arange(1,mK+1)],
                indices=tt,
                n_cores=cores
            )
            toc = time.time()
            print(f'KNN time: {toc-tic}', end='\r')
            knn_t += toc-tic

            cv_scores_svm.append(svm_scores)
            cv_scores_knn.append(knn_scores)

        
    cv_scores_svm = pd.concat(cv_scores_svm, ignore_index=True)
    cv_scores_knn = pd.concat(cv_scores_knn, ignore_index=True)

    # add column with param keys as dict
    cv_scores_svm['degree'] = cv_scores_svm['degree'].astype('Int64')
    cv_scores_svm['params'] = cv_scores_svm.apply(
        lambda x: str({k: x[k] for k in param_keys if not pd.isna(x[k])}), axis=1
    )
    cv_scores_knn['params'] = cv_scores_knn['n_neighbors'].apply(
        lambda x: str({'n_neighbors': x})
    )

    cv_scores_knn[['proj','params','score']].to_csv(outpath+'knn_scores.csv')
    cv_scores_svm[['proj','params','score']].to_csv(outpath+'svm_scores.csv')
    
    mean_scores_svm = cv_scores_svm.groupby(['proj','params'])[['score']].aggregate(['mean','max','std']).reset_index()
    mean_scores_knn = cv_scores_knn.groupby(['proj','params'])[['score']].aggregate(['mean','max','std']).reset_index()

    best_svm = mean_scores_svm.sort_values([('score','mean'),('score','max'),('score','std')], 
                            ascending=[False,False,True]).groupby('proj').head(1)
    best_knn = mean_scores_knn.sort_values([('score','mean'),('score','max'),('score','std')], 
                            ascending=[False,False,True]).groupby('proj').head(1)

    pred_params = pd.DataFrame({'SVM':best_svm.set_index('proj')['params'].apply(eval),
                'KNN':best_knn.set_index('proj')['params'].apply(eval)})
    
    pred_params.to_csv(outpath+'pred_params.csv',index_label='proj')

    return pred_params

def pred_cv_basic(metadata, selFeats, outpath, iSpace):
    '''metadata should be training data only'''

    train_ind = list(metadata.index)
    
    cores = max(iSpace.parallel,1) if iSpace is not None else 1

    # skip DA proj
    da_list = [p for p in iSpace.PlotProj.keys() if "DA" in p]
    for proj in da_list:
        iSpace.delProj(proj)
    
    def generate_params_(min, max):  

        rbf = [{'C': 10**i, 'gamma': 10**j, 'kernel': 'rbf'} 
                    for i in range(min, max+1) for j in range(min, max+1)]
        
        # poly = [{'C': 10**i, 'degree': j, 'kernel': 'poly'} 
        #             for i in range(min, max+1) for j in range(1,4)]
        
        return  rbf #+ poly    
            
    
    param_space = generate_params_(-3,3)

    param_keys = [p.keys() for p in param_space]
    param_keys = list(set().union(*param_keys))  # Unique keys across all param sets

    mK = round(len(train_ind)*0.05)
    
    cv_scores_svm = []
    cv_scores_knn = []

    gen_seed = np.random.default_rng(1111)

    # repeated cross-validation

    for f in range(2):
        print(f'Repeat {f}')
        
        cv = [inds for inds in StratifiedKFold(n_splits=5, shuffle=True, random_state=gen_seed.integers(1,100)).split(
            metadata[selFeats], metadata['Best']
        )]

        for i in range(5):
            print(f'Fold {i}')
            cv_train = [train_ind[j] for j in cv[i][0]]
            cv_test = [train_ind[j] for j in cv[i][1]]
            tt = (np.arange(len(cv_train)), np.arange(len(cv_train),len(cv_train)+len(cv_test)))
            
            svm_t = 0
            knn_t = 0

            #### SVM CV
            tic = time.time()
            svm_scores = grid_search_cv(
                estimator=SVC(random_state=1111,probability=False,coef0=1),
                iSpace=iSpace, param_grid=param_space, indices=tt, 
                n_cores=cores, basic=True
            )
            toc = time.time()
            print(f'SVM time: {toc-tic}', end='\r')
            svm_t += toc-tic

            #### knn CV
            tic = time.time()
            knn_scores = grid_search_cv(
                estimator=KNeighborsClassifier(),
                iSpace=iSpace,
                param_grid=[{'n_neighbors': p} for p in np.arange(1,mK+1)],
                indices=tt,
                n_cores=cores, basic=True
            )
            toc = time.time()
            print(f'KNN time: {toc-tic}', end='\r')
            knn_t += toc-tic

            cv_scores_svm.append(svm_scores)
            cv_scores_knn.append(knn_scores)

        
    cv_scores_svm = pd.concat(cv_scores_svm, ignore_index=True)
    cv_scores_knn = pd.concat(cv_scores_knn, ignore_index=True)

    # add column with param keys as dict
    cv_scores_svm['degree'] = cv_scores_svm['degree'].astype('Int64')
    cv_scores_svm['params'] = cv_scores_svm.apply(
        lambda x: str({k: x[k] for k in param_keys if not pd.isna(x[k])}), axis=1
    )
    cv_scores_knn['params'] = cv_scores_knn['n_neighbors'].apply(
        lambda x: str({'n_neighbors': x})
    )

    # cv_scores_knn[['proj','params','score']].to_csv(outpath+'knn_scoresB.csv')
    # cv_scores_svm[['proj','params','score']].to_csv(outpath+'svm_scoresB.csv')
    
    mean_scores_svm = cv_scores_svm.groupby(['proj','params'])[['score']].aggregate(['mean','max','std']).reset_index()
    mean_scores_knn = cv_scores_knn.groupby(['proj','params'])[['score']].aggregate(['mean','max','std']).reset_index()

    best_svm = mean_scores_svm.sort_values([('score','mean'),('score','max'),('score','std')], 
                            ascending=[False,False,True]).groupby('proj').head(1)
    best_knn = mean_scores_knn.sort_values([('score','mean'),('score','max'),('score','std')], 
                            ascending=[False,False,True]).groupby('proj').head(1)

    pred_params = pd.DataFrame({'SVM':best_svm.set_index('proj')['params'].apply(eval),
                'KNN':best_knn.set_index('proj')['params'].apply(eval)})
    
    pred_params.to_csv(outpath+'pred_paramsB.csv',index_label='proj')

    return pred_params





if __name__ == "__main__":
    
    # outpath = '../Results/tsp/'
    # iSpace = pd.read_pickle(outpath+'iSpace.pkl')

    # is_eval = InstanceSpaceEval(iSpace, split=True)
    # is_eval.proj_evaluation()
    # with open(outpath+'proj_eval.pkl','wb') as f:
    #     pkl.dump(is_eval, f)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('outpath', type=str)
    parser.add_argument('cores', type=int)

    args = parser.parse_args()
    outpath = args.outpath
    cores = args.cores

    
    selFeats = list(pd.read_csv(outpath + 'selected_features.csv', index_col=0).iloc[0].values)
    metadata_train = pd.read_csv(f'{outpath}metadata_train.csv', index_col=0)

    with open(outpath + 'iSpace.pkl', 'rb') as f:
        iSpace = pkl.load(f)

    iSpace.doParallel(cores)
    pred_paramDF = pred_cv_basic(metadata_train,selFeats,outpath,iSpace)