import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.optimize import fmin_bfgs
# from scipy import stats
# from scipy.linalg import eig
from sklearn.metrics import mean_squared_error

import multiprocessing as mp

# import pyispace

# import matplotlib.pyplot as plt
# import seaborn as sns

class Pilot():

    def __init__(self):
        self.V = []
        self.A = []
        self.B = []
        self.C = []

        # after dropping dependent features
        self.F_ = []
        self.A_ = []
        self.dropped = []
        self.d_max = 0
        

    def drop_dependent(self,X):
        Rank = np.linalg.matrix_rank(X)
        dropped = []
        featInd = np.arange(X.shape[1])
        
        i=X.shape[1]-1
        while i > 0 and X.shape[1] > Rank:
            rank = np.linalg.matrix_rank(X)
            if np.linalg.matrix_rank(np.delete(X, i, axis=1)) == rank:
                dropped.append(featInd[i])
                featInd = np.delete(featInd, i)
                X = np.delete(X, i, axis=1)
            i -= 1

        return sorted(dropped), X

    def fitExtra(self, F, Y, d=0, num_params={},
            featlabels=None, alglabels=None):  
        # added conditions for convergence of numerical solution      
            
        Xbar = np.hstack((F, Y))
        m = F.shape[1]
        a = Y.shape[1]

        F = F.T
        Xbar = Xbar.T
        # add names to Xbar

        # analytical solution
        if len(num_params) == 0:
            D,V = np.linalg.eigh(np.dot(Xbar, Xbar.T))  

            # D is automatically cast to real, but V is not
            if np.isrealobj(D):
                V = V.real
            else:
                print('Warning: V is complex')
            self.d_max = len(D)

            idx = np.argsort(np.abs(D))[::-1]
            if d == 0:
                d = len(idx)
            V = V[:, idx[:d]]          # top d eigenvectors
            self.B = V[:m, :]            # rows for features
            self.C = V[m:, :]          # rows for performance

            Xr = np.dot(F.T, np.linalg.pinv(np.dot(F, F.T)))
            A = np.dot(V.T, np.dot(Xbar, Xr))
                    
            self.V = V
            self.A = A

        else:
        # numerical solution
            if d == 0:
                print('Error: d must be specified for numerical solution - doing 2D projection')
                d = 2
            rng = np.random.default_rng(num_params['seed'])

            def errorfun(X,F,Y,m,a,d):
                A = np.reshape(X[:m*d], (d,m))
                B = np.reshape(X[m*d:m*d+m*d], (m,d))
                C = np.reshape(X[m*d+m*d:], (a,d))

                Z = np.dot(A, F)
                Fhat = np.dot(B, Z).T
                Yhat = np.dot(C, Z).T
                f_err = mean_squared_error(F.T, Fhat)
                y_err = mean_squared_error(Y, Yhat)
                return f_err + y_err

            D_high = distance.pdist(F.T, 'euclidean')
            sol = {'rho':np.inf, 'f':np.inf,
                   'A':[],'B':[],'C':[]}
            
            tries = 0

            for i in range(num_params['ntries']):
                tries += 1
                X0 = rng.uniform(-1,1,size=(m*d+m*d+a*d))

                out_i = fmin_bfgs(errorfun,X0,args=(F,Y,m,a,d), 
                               full_output=True, disp=False)
                Xi = out_i[0]
                fi = out_i[1]

                while tries > 1 and fi < sol['f'] and tries < num_params['maxtries']:
                    print(f'Warning: f is decreasing - adding more tries {tries}')
                    tries += 1
                    sol['rho'] = np.inf

                    X0 = rng.uniform(-1,1,size=(m*d+m*d+a*d))
                    out_i = fmin_bfgs(errorfun,X0,args=(F,Y,m,a,d), 
                               full_output=True, disp=False)
                    Xi = out_i[0]
                    fi = out_i[1]
                    
                A = np.reshape(Xi[:m*d], (d,m))
                B = np.reshape(Xi[m*d:m*d+m*d], (m,d))
                C = np.reshape(Xi[m*d+m*d:], (a,d))

                Z = np.dot(A, F)
                D_low = distance.pdist(Z.T, 'euclidean')
                rho = np.corrcoef(D_high, D_low)[0,1]
                if rho < sol['rho']:
                    sol['rho'] = rho
                    sol['A'] = A
                    sol['B'] = B
                    sol['C'] = C
                    sol['Z'] = Z

            self.A, self.B, self.C = sol['A'], sol['B'], sol['C']

    def _fit_analytical(self, F, Xbar, m, a, d):
        D,V = np.linalg.eigh(np.dot(Xbar, Xbar.T))  

        # D is automatically cast to real, but V is not
        if np.isrealobj(D):
            V = V.real
        else:
            print('Warning: V is complex')
        self.d_max = len(D)

        idx = np.argsort(np.abs(D))[::-1]
        if d == 0:
            d = len(idx)
        V = V[:, idx[:d]]          # top d eigenvectors
        self.B = V[:m, :]            # rows for features
        self.C = V[m:, :]          # rows for performance

        Xr = np.dot(F.T, np.linalg.pinv(np.dot(F, F.T)))
        A = np.dot(V.T, np.dot(Xbar, Xr))
                
        self.V = V
        self.A = A

    def _fit_numerical(self, seed, F, Y, d, m, a, D_high):

        def errorfun(X,F,Y,m,a,d):
            A = np.reshape(X[:m*d], (d,m))
            B = np.reshape(X[m*d:m*d+m*d], (m,d))
            C = np.reshape(X[m*d+m*d:], (a,d))

            Z = np.dot(A, F)
            Fhat = np.dot(B, Z).T
            Yhat = np.dot(C, Z).T
            f_err = mean_squared_error(F.T, Fhat)
            y_err = mean_squared_error(Y, Yhat)
            return f_err + y_err
        
        rng = np.random.default_rng(seed) 
        X0 = rng.uniform(-1,1,size=(m*d+m*d+a*d))
        
        Xi = fmin_bfgs(errorfun,X0,args=(F,Y,m,a,d), 
                            full_output=False, disp=False)
        A = np.reshape(Xi[:m*d], (d,m))
        B = np.reshape(Xi[m*d:m*d+m*d], (m,d))
        C = np.reshape(Xi[m*d+m*d:], (a,d))

        Z = np.dot(A, F)
        D_low = distance.pdist(Z.T, 'euclidean')
        rho = np.corrcoef(D_high, D_low)[0,1]

        return {'rho': rho, 'A': A, 'B': B, 'C': C}            

    def fit(self, F, Y, d=0, num_params={},
        featlabels=None, alglabels=None):        
        
        Xbar = np.hstack((F, Y))
        m = F.shape[1]
        a = Y.shape[1]

        F = F.T
        Xbar = Xbar.T
        # add names to Xbar

        # analytical solution
        if len(num_params) == 0:
            self._fit_analytical(F, Xbar, m, a, d)
            
        else:
        # numerical solution
            if d == 0:
                print('Error: d must be specified for numerical solution - doing 2D projection')
                d = 2
            
            D_high = distance.pdist(F.T, 'euclidean')
            
            ## have to use seed sequence
            ## for reproducibility between sequential and parallel runs
            ss = np.random.SeedSequence(num_params['seed'])
            child_seqs = ss.spawn(num_params['ntries'])
            seed_ints = [cs.generate_state(1)[0] for cs in child_seqs]

            if 'parallel' not in num_params.keys() or not isinstance(num_params['parallel'], int):
                sols = [self._fit_numerical(s, F, Y, d, m, a, D_high) for s in seed_ints]
            
            else:
                n_cores = num_params['parallel']                
            
                pool = mp.Pool(n_cores)
                results = [pool.apply_async(self._fit_numerical,
                            args=(s, F, Y, d, m, a, D_high)) for s in seed_ints]
                
                pool.close()
                pool.join()

                sols = [res.get() for res in results]
            
            # find the best solution - max rho
            best_sol = max(sols, key=lambda x: x['rho'])

            self.A, self.B, self.C = best_sol['A'], best_sol['B'], best_sol['C']
            
        
   
    def fit_drop(self, F, Y, d=0,featlabels=None, alglabels=None):
        
        if np.linalg.matrix_rank(F) < F.shape[1]:
            self.dropped, F = self.drop_dependent(F)
            
        Xbar = np.hstack((F, Y))
        m = F.shape[1]
        a = Y.shape[1]

        F = F.T
        Xbar = Xbar.T
        # add names to Xbar


        D,V = np.linalg.eig(np.dot(Xbar, Xbar.T))  
        if np.all(D.imag == 0):
            D = D.real
            V = V.real
        self.d_max = len(D)

        idx = np.argsort(np.abs(D))[::-1]
        if d == 0:
            d = len(idx)
        V = V[:, idx[:d]]          # top d eigenvectors
        self.B = V[:m, :]            # rows for features
        self.C = V[m:, :]          # rows for performance

        Xr = np.dot(F.T, np.linalg.pinv(np.dot(F, F.T)))
        A = np.dot(V.T, np.dot(Xbar, Xr))
        #Z = np.dot(A, F)
        #Xhat = np.vstack((np.dot(B, Z), np.dot(C, Z)))
        #error = float(np.sum((Xbar - Xhat) ** 2))
        # R2 = np.diagonal(np.corrcoef(Xbar, Xhat, rowvar=False)[:m+a, m+a:]) ** 2
        #Z = Z.T

        self.A_ = A.copy()
        self.dropped.sort()
        for c in self.dropped:
            A = np.insert(A, c, 0, axis=1)

        self.V = V
        self.A = A
        self.F_ = F.T
        
    def transform(self, F):
        return np.dot(self.A, F.T).T
    
    def error0(self, F, Y):
        
        F_ = np.delete(F, self.dropped, axis=1)
        
        Z = np.dot(self.A, F.T)
        Fhat = np.dot(self.B, Z).T
        Yhat = np.dot(self.C, Z).T

        f_err = mean_squared_error(F_, Fhat)
        y_err = mean_squared_error(Y, Yhat)
        
        return {'loss': f_err + y_err, 'F loss': f_err, 'Y loss': y_err}

    def error_split(self, F, Y, d):
        # F = self.scaleF.transform(F)
        # Y = self.scaleY.transform(Y)

        F_ = np.delete(F, self.dropped, axis=1)
        
        Z = np.dot(self.A[:d,:], F.T)
        Fhat = np.dot(self.B[:,:d], Z).T
        Yhat = np.dot(self.C[:,:d], Z).T
        
        f_err = mean_squared_error(F_, Fhat)
        y_err = mean_squared_error(Y, Yhat)
        
        # return np.sum((F_.T - Fhat) ** 2)/Fhat.shape[1], np.sum((Y.T - Yhat) ** 2)/Yhat.shape[1]
        return {'loss': f_err + y_err, 'F loss': f_err, 'Y loss': y_err}

def pilot_CV(X,Y,skf,scale,return_best, return_errors):

    train_errors = []
    test_errors = []
    
    for train_idx, test_idx in skf:
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        if scale:
            scaleX = StandardScaler().fit(Xtrain)
            scaleY = StandardScaler().fit(Ytrain)
            Xtrain = scaleX.transform(Xtrain)
            Ytrain = scaleY.transform(Ytrain)
            Xtest = scaleX.transform(Xtest)
            Ytest = scaleY.transform(Ytest)

        pilot = Pilot()
        pilot.fit(Xtrain, Ytrain, d=0)

        train_errors.append(
            {d:pilot.error_split(Xtrain, Ytrain, d)['loss'] for d in range(1,pilot.d_max+1)}
        )
        test_errors.append(
            {d:pilot.error_split(Xtest, Ytest, d)['loss'] for d in range(1,pilot.d_max+1)}
        )

    train_errors = pd.DataFrame(train_errors).mean()
    test_errors = pd.DataFrame(test_errors).mean()

    if return_best:
        best_d = test_errors.idxmin()

        if scale:
            scaleX = StandardScaler().fit(X)
            scaleY = StandardScaler().fit(Y)
            X = scaleX.transform(X)
            Y = scaleY.transform(Y)

        pilotB = Pilot()
        pilotB.fit(X, Y, d=best_d)
        if return_errors:
            return {'model':pilotB, 'best_d':best_d,'train':train_errors, 'test':test_errors}
        return {'model':pilotB, 'best_d':best_d}    
    return {'train':train_errors, 'test':test_errors}
    



if __name__ == '__main__':
    feature_data = pd.read_csv('./NL/feature_process.csv')
    feature_matrix = feature_data.iloc[:, 1:].to_numpy()
    algorithm_data = pd.read_csv('./NL/algorithm_process.csv')
    algorithm_matrix = algorithm_data.iloc[:, 1:].to_numpy() 

    pilotN = Pilot()
    pilotN.fit(feature_matrix, algorithm_matrix, d=2,
               num_params={'ntries':2, 'seed':111, 'maxtries':10})