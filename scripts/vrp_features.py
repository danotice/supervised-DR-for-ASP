import vrplib

import numpy as np
import pandas as pd
import scipy.spatial 
import scipy.stats
import scipy.sparse.csgraph
import math
from sklearn.cluster import DBSCAN
import gurobipy as gp
from gurobipy import GRB


import matplotlib.pyplot as plt

import time
from copy import copy
import os
import multiprocessing as mp

# helper functions
def numvec_feature_statistics(x, name, set='a', na_rm=True):
    x = np.array(x)
    if na_rm:
        x = x[~np.isnan(x)]
    
    xmean = np.mean(x)
    xsd = np.std(x)
    out_dict = {f'{name}_mean': xmean,
        f'{name}_sd': xsd,
        f'{name}_CV': xsd / xmean if xmean != 0 else np.nan
    }

    if set != 'a':
        xmin = np.min(x)
        xmax = np.max(x)
    
        out_dict[f'{name}_min'] = xmin
        out_dict[f'{name}_max'] = xmax
        out_dict[f'{name}_range'] = xmax - xmin
        out_dict[f'{name}_median'] = np.median(x)

        if set == 'c':
            out_dict[f'{name}_skew'] = scipy.stats.skew(x)
            out_dict[f'{name}_kurt'] = scipy.stats.kurtosis(x)   

    
    return out_dict

def numvec_mode_description(x, name, na_rm=True):
    # only returns the first mode
    x = np.array(x)
    if na_rm:
        x = x[~np.isnan(x)]

    modes = scipy.stats.mode(x, keepdims=True)

    out_dict = {
        f'{name}_mode_freq': modes.count,
        f'{name}_mode_value': modes.mode,
    }

    return out_dict

def angle_between_points(a, b, c):
    # TODO - change so no more runtime warnings
    v1 = (a - b) / np.linalg.norm(a - b)
    v2 = (a - c) / np.linalg.norm(a - c)
    z = np.dot(v1, v2)
    
    if 1 - z < 1e-10:
        angle = 0
    elif z - (-1) < 1e-10:
        angle = math.pi
    else:
        angle = math.acos(z)
    
    return min(abs(angle + np.array([0, 2 * math.pi, -2 * math.pi])))



class CVRP:

    def __init__(self, name, instance):
        # takes a VRP instance as dict

        self.name = name
        self.n = instance['dimension'] - 1      # number of customers
        self.demands = instance['demand']   # demands of customers - should include 0 for depot
        self.capacity = instance['capacity']

        self.dist = instance['edge_weight']     # includes distance to depot
        self.norm_dists = None

        #coordinates
        self.coords = instance['node_coord']
        self.depot_ind = int(instance['depot'][0])
        if self.depot_ind != 0:
            raise ValueError('Depot index is not 0.')
            # eventually move depot to index 0

        self.depot_coords = self.coords[self.depot_ind]
        self.norm_coords = None
        self.norm_depot_coords = None
        self.norm_scale = None

        self.k_min = instance['k_min']
        

    @classmethod
    def from_file(cls, dir, name, k_known):
        # reads a VRP instance from a file

        if dir[-1] != '/':
            dir = dir + '/'
        instance = vrplib.read_instance(dir + name +'.vrp')

        if k_known==True:
            instance['k_min'] = int(name.split('k')[-1])
                       
        elif type(k_known)==int:
            instance['k_min'] = k_known
            
        else:
            instance['k_min'] = None
            

        return cls(name, instance)

    def normalise_instance(self,scale):

        self.norm_scale = scale
         
        min = np.min(self.coords)
        max = np.max(self.coords)
        self.norm_coords = (self.coords - min) / (max - min) * scale
        self.norm_dists = self.dist / (max - min) * scale
        self.norm_depot_coords = self.norm_coords[self.depot_ind]

        #print(f'Normalised instance to range {scale}.')

    def calc_k_min(self, simp=False):
        # calculate the minimum number of vehicles required to serve all customers
        # using the bin packing formulation

        if simp:
            self.k_min = int(np.ceil(np.sum(self.demands)/self.capacity))

        else:
            cap = self.capacity
            loads = [l for l in self.demands if l > 0]
            n = len(loads)

            model = gp.Model()
            model.Params.LogToConsole = 0

            x = model.addVars(n, n, vtype=GRB.BINARY)
            y = model.addVars (n, vtype=GRB.BINARY)

            # minimize the number of bins used
            model.setObjective(gp.quicksum(y[j] for j in range(n) ), GRB.MINIMIZE)

            # pack each item in exactly one bin
            model.addConstrs(gp.quicksum(x[i,j] for j in range(n)) == 1 for i in range(n))

            # bin capacity constraint
            model.addConstrs(gp.quicksum(loads[i] * x[i,j] for i in range(n)) <= cap * y[j] for j in range (n))

            # solve
            model.optimize()
            if model.Status == 2:
                self.k_min = int(model.ObjVal)
            else:
                print('Could not calculate k_min.')
        
    def features_distance(self, norm = False, all = False):
        """Distance matrix features (set C stats)

        Args:
            norm (bool, optional): calculate on normalised instance. Defaults to False.
            all (bool, optional): calculate mode stats. Defaults to False.

        Returns:
            dict
        """
        # returns a dict of summary stats of the distance matrix
        if norm:
            dist = scipy.spatial.distance.squareform(self.norm_dists[1:,1:]) 
            label = 'norm_dist'
        else:
            dist = scipy.spatial.distance.squareform(self.dist[1:,1:])
            label = 'dist'

        out_dict = numvec_feature_statistics(dist, label, set='c')

        if all: 
            out_dict[label+'_lt_mean'] = np.mean(dist < np.mean(dist))

            dist_distinct = np.unique(np.around(dist,2), return_counts=True)
            out_dict[label+'_distinct'] = len(dist_distinct)/len(dist)
            out_dict[label+'_mode_freq'] = np.max(dist_distinct[1])
            mode_value = dist_distinct[0][dist_distinct[1]== out_dict[label+'_mode_freq']]

            out_dict[label+'_mode_value'] = np.mean(mode_value) 
            out_dict[label+'_mode_quantity'] = len(mode_value)
        
        return out_dict
        
    def features_depot(self, norm = False):

        # returns a dict of stats of distance to depot
        if norm:
            dist = self.norm_dists[1:,0]
            depot = self.norm_depot_coords
        else:
            dist = self.dist[1:,0]
            depot = self.depot_coords

        out_dict = numvec_feature_statistics(dist, 'depot_dist', set='c')
        out_dict['depot_coord_X'] = depot[0]
        out_dict['depot_coord_Y'] = depot[1]

        if self.k_min is not None:
            nn_k = np.sort(dist)[0:self.k_min]
            out_dict.update(numvec_feature_statistics(nn_k, 'depot_nn', set='a'))
        
        return out_dict

    def features_centroid(self, norm = True):

        if norm:
            coords = self.norm_coords[1:]
            depot_coords = self.norm_depot_coords            
        else:
            coords = self.coords[1:]  
            depot_coords = self.depot_coords          
        
        centroid = np.mean(coords, axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)

        out_dict = numvec_feature_statistics(dists, 'dist_centroid', set='c')
        out_dict['centroid_coord_X'] = centroid[0]
        out_dict['centroid_coord_Y'] = centroid[1]
        out_dict['depot_centroid_dist'] = np.linalg.norm(centroid - depot_coords)

        return out_dict
    
    def features_demand(self):
        # returns a dict of summary stats of the demand vector 
        # normalized by capacity

        demands = self.demands[self.demands>0]
        norm_demand = demands/self.capacity
        total_demand = np.sum(demands)

        out_dict = numvec_feature_statistics(norm_demand, 'demand', set='b')
        
        if self.k_min is not None:            
            out_dict['demand_to_capacity'] = total_demand/(self.k_min*self.capacity)
            out_dict['min_vehicles_gap'] = self.k_min - np.ceil(total_demand/self.capacity)
            out_dict['customers_to_vehicle'] = self.n/self.k_min

        return out_dict


    def features_nearest_neighbours(self, depot, norm = True):
        # if depot = True, nearest neighbours can be depot
        if norm:
            dist = copy(self.norm_dists) # copy because of fill_diagonal later
            coords = self.norm_coords
        else:
            dist = copy(self.dist)
            coords = self.coords
        
        np.fill_diagonal(dist, np.inf) # so self is not nearest neighbour
        nn_depot = np.mean(np.argmin(dist[1:,], axis=1)==0)

        if not depot:
            dist = dist[1:,1:] # remove depot row and column
            coords = coords[1:]
        #else:
            #dist = dist[1:,:] # remove depot row regardless. keep column
            #coords = coords[1:]
            
        neighbours = np.argsort(dist, axis=1)[:,0:2] # coord indices of nearest neighbours
        nnd1 = np.min(dist, axis=1) # nearest neighbour distances

        angles = np.array(
            [angle_between_points(coords[i], coords[neighbours[i,0]], coords[neighbours[i,1]]) for i in range(len(coords))])

        out_dict = numvec_feature_statistics(nnd1, 'nn_dist', set='c')
        out_dict['nn_depot_prop'] = nn_depot
        out_dict.update(numvec_feature_statistics(angles, 'nn_angle', set='b'))

        return out_dict

    def features_minimum_spanning_tree(self, norm = False):

        if norm:
            dist = self.norm_dists 
        else:
            dist = self.dist

        mst = scipy.sparse.csgraph.minimum_spanning_tree(dist)
        mst = mst.toarray()

        mst_dists = mst[mst>0]

        out_dict = numvec_feature_statistics(mst_dists, 'mst_dists', set='b')
        out_dict['mst_dist_len'] = len(mst_dists)
        out_dict['mst_dist_sum'] = np.sum(mst_dists)/np.sum(dist) #maybe divide by 2


        # node degrees
        mst_degree = np.sum(mst > 0,axis=0) + np.sum(mst > 0,axis=1)        
        out_dict.update(numvec_feature_statistics(mst_degree, 'mst_degree', set='b'))
        
        return out_dict

    def features_geometric(self, norm = True):

        out_dict = dict()

        if norm:
            coords = self.norm_coords
            dist = self.norm_dists
        else:
            coords = self.coords
            dist = self.dist

        out_dict['rect_area'] = (np.max(coords[:,0]) - np.min(coords[:,0])) * (np.max(coords[:,1]) - np.min(coords[:,1]))

        hull = scipy.spatial.ConvexHull(coords)
        out_dict['chull_area'] = hull.volume
        out_dict['chull_perimeter'] = hull.area
        out_dict['chull_points_frac'] = len(hull.vertices)/len(coords)
        out_dict['chull_depot'] = self.depot_ind in hull.vertices

        # edges on convex hull
        chull_edges = np.array([dist[i,j] for [i,j] in hull.simplices])
        out_dict.update(numvec_feature_statistics(chull_edges, 'chull_edges', set='a'))

        # distances to hull contour
        #chull_dists = np.array([scipy.spatial.distance.euclidean(coords[i], coords[j]) for [i,j] in hull.simplices])


        #plot convex hull
        # plt.scatter(coords[:,0],coords[:,1])
        # for s in hull.simplices:
        #     plt.plot(coords[s,0], coords[s,1])

        return out_dict
    
    def features_clustering(self, depot = True, norm = True, 
                            min_cluster_size = 4,eps = 'u', plot = False):

        out_dict = dict()

        if norm:
            coords = self.norm_coords
            dist = self.norm_dists
        else:
            coords = self.coords
            dist = self.dist

        if not depot:
            coords = coords[1:]
            dist = dist[1:,1:]

        if eps == 'u':
            epsilon = self.norm_scale/(math.sqrt(self.n))   
        else:
            epsilon = np.percentile(dist[dist>0],eps)

        db = DBSCAN(eps=epsilon, min_samples=min_cluster_size, metric='precomputed')
        clusters = db.fit_predict(dist)
        clus_counts = np.unique(clusters, return_counts=True)        

        n = len(clusters)
        
        if clus_counts[0][0] == -1: # if there are outliers
            out_dict['cluster_count'] = len(clus_counts[0])-1
            out_dict['cluster_outlier_prop'] = clus_counts[1][0]/n

            if out_dict['cluster_count'] == 0: # if all points are outliers, treat as one cluster
                clust_size = clus_counts[1]
                clust_coords = {i:coords[clusters==i] for i in clus_counts[0]}
            else:
                clust_size = clus_counts[1][1:]
                clust_coords = {i:coords[clusters==i] for i in clus_counts[0][1:]}

        else:
            out_dict['cluster_count'] = len(clus_counts[0])
            out_dict['cluster_outlier_prop'] = 0
            clust_size = clus_counts[1]
            clust_coords = {i:coords[clusters==i] for i in clus_counts[0]}

        if self.k_min is not None:
            out_dict['cluster_count_rel_vehicle'] = out_dict['cluster_count']/self.k_min
        
        out_dict['cluster_count_rel_nodes'] = out_dict['cluster_count']/n

        out_dict['cluster_core_prop'] = len(db.core_sample_indices_)/n
        out_dict['cluster_edge_prop'] = 1 - out_dict['cluster_core_prop'] - out_dict['cluster_outlier_prop']

        out_dict.update(numvec_feature_statistics(clust_size, 'cluster_size', set='a'))

        # cluster centroids
        centroids = np.array([np.mean(clust_coords[i], axis=0) for i in clust_coords.keys()])
        dists = np.concatenate([np.linalg.norm(clust_coords[i] - centroids[i], axis=1) for i in clust_coords.keys()])

        out_dict.update(numvec_feature_statistics(dists, 'cluster_centroid_dist', set='b'))

        if plot:
            self.plot_instance(norm=norm, colour=clusters)

        return out_dict


    def plot_instance(self, norm = True, colour = None):
        
        if norm:
            coords = self.norm_coords
        else:
            coords = self.coords

        plt.scatter(coords[:,0],coords[:,1], c = colour)
        plt.scatter(coords[self.depot_ind,0],coords[self.depot_ind,1], color='black', marker='x', alpha=0.7)
        plt.show()


def get_features(ins, folder, k, norm_size=400, calc_k = True, simp_k = False,printc = True):
    # test features on a single instance
    # ins is a CVRP instance name

    tic = time.perf_counter()
    eg = CVRP.from_file(folder, ins, k_known=k)

    eg.normalise_instance(norm_size)
    toc = time.perf_counter()
    if printc:
        print(f'Read and normalised instance {eg.name} in {toc - tic:0.4f} seconds')

    tic = time.perf_counter()

    if eg.k_min is None and calc_k:
        tic = time.perf_counter()
        eg.calc_k_min(simp_k)
        toc = time.perf_counter()

        if printc and not simp_k:
            print(f'Calculated {eg.name} k_min in {toc - tic:0.4f} seconds')
            

    D = eg.features_distance(norm=True,all=True)
    Dp = eg.features_depot(norm=True)
    C = eg.features_centroid(norm=True)
    Dem = eg.features_demand()
    N = eg.features_nearest_neighbours(depot=True, norm=True)
    M = eg.features_minimum_spanning_tree(norm=True)
    Ch = eg.features_geometric(norm=True)
    Cl = eg.features_clustering(depot=False, norm=True, min_cluster_size=4, eps='u', plot=False)    

    toc = time.perf_counter()

    k_lab = 'k_min_' if simp_k else 'k_min'
    
    features = pd.DataFrame({
        **{'instance': eg.name, 'n': eg.n, k_lab: eg.k_min, 'capacity': eg.capacity},
        **D, **Dp,**C, **Dem, **N, **M, **Ch, **Cl}, 
        index=[0])

    if printc:
        print(f'{eg.name} Extracted {features.shape[1]-1} features in {toc - tic:0.4f} seconds \n')
        
    return eg, features

def get_features_from_folder(folder, k, norm_size, save):
    # get features from all instances in a folder
    # folder is a path to a folder containing .vrp files

    files = [f for f in os.listdir(folder) if f.endswith('.vrp')]
    N = len(files)

    features = []

    for (i,f) in enumerate(files):
        print(f'{i+1}/{N} Extracting features from {f}...') #, end = '\r')
        _, fts = get_features(f.split('.')[0],folder, k, norm_size, calc_k=True, printc=False)
        features.append(fts)
    
    features = pd.concat(features, ignore_index=True)
    features.set_index('instance', inplace=True)
    features.sort_values('n', inplace=True)
    features.to_csv(save)

def get_features_from_folder_p(cores, folder, k, simp_k ,norm_size, save):
    # get features from all instances in a folder
    # folder is a path to a folder containing .vrp files

    files = [f for f in os.listdir(folder) if f.endswith('.vrp')]
    
    pool = mp.Pool(cores)
    results = [pool.apply_async(get_features, 
                args=(f.split('.')[0],folder, k, norm_size, True, simp_k, True)) for f in files]
    
    pool.close()
    pool.join()

    features = [r.get()[1] for r in results]    
    features = pd.concat(features, ignore_index=True)
    features.set_index('instance', inplace=True)
    features.sort_values('n', inplace=True)
    features.to_csv(save)

def get_dist(ins, folder, k, norm_size=400, printc = True):
    # test features on a single instance
    # ins is a CVRP instance name

    tic = time.perf_counter()
    eg = CVRP.from_file(folder, ins, k_known=k)

    eg.normalise_instance(norm_size)
    toc = time.perf_counter()
    if printc:
        print(f'Read and normalised instance {eg.name} in {toc - tic:0.4f} seconds')

    tic = time.perf_counter()

    min = np.min(eg.coords)
    max = np.max(eg.coords)
    D = eg.features_distance(norm=False,all=True)
    
    features = pd.DataFrame({**{'instance': eg.name, 'n': eg.n, 'norm_scale': max - min},**D}, 
        index=[0])

        
    return eg, features

def get_dist_from_folder_p(cores, folder, k, norm_size, save):
    # get features from all instances in a folder
    # folder is a path to a folder containing .vrp files

    files = [f for f in os.listdir(folder) if f.endswith('.vrp')]#[:3]
    
    pool = mp.Pool(cores)
    results = [pool.apply_async(get_dist, 
                args=(f.split('.')[0],folder, k, norm_size, True)) for f in files]
    
    pool.close()
    pool.join()

    features = [r.get()[1] for r in results]    
    features = pd.concat(features, ignore_index=True)
    features.set_index('instance', inplace=True)
    features.sort_values('n', inplace=True)
    features.to_csv(save)


def get_features0(ins, folder = './Instances', norm_size=400, printc = True):
    # test features on a single instance
    # ins is a CVRP instance name

    tic = time.perf_counter()
    eg = CVRP.from_file(folder, ins, k_known=True)

    eg.normalise_instance(norm_size)
    toc = time.perf_counter()
    if printc:
        print(f'Read and normalised instance {eg.name} in {toc - tic:0.4f} seconds')

    tic = time.perf_counter()

    D = eg.features_distance(norm=False,all=True)
    Dn = eg.features_distance(norm=True, all=False)
    Dp = eg.features_depot(norm=True)
    C = eg.features_centroid(norm=True)
    Dem = eg.features_demand()
    N = eg.features_nearest_neighbours(depot=True, norm=True)
    M = eg.features_minimum_spanning_tree(norm=True)
    Ch = eg.features_geometric(norm=True)
    Cl = eg.features_clustering(depot=False, norm=True, min_cluster_size=4, eps='u', plot=False)    

    toc = time.perf_counter()
    
    features = pd.DataFrame({
        **{'instance': eg.name, 'n': eg.n, 'k_min': eg.k_min, 'capacity': eg.capacity},
        **D, **Dn, **Dp,**C, **Dem, **N, **M, **Ch, **Cl}, 
        index=[0])

    if printc:
        print(f'Extracted {features.shape[1]-1} features in {toc - tic:0.4f} seconds')
        print()

    return eg, features

def get_features_from_folder0(folder, norm_size):
    # get features from all instances in a folder
    # folder is a path to a folder containing .vrp files

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files = [f for f in files if f.endswith('.vrp')]
    N = len(files)

    features = pd.DataFrame()

    for (i,f) in enumerate(files):
        print(f'{i+1}/{N} Extracting features from {f}...', end = '\r')
        _, fts = get_features0(f.split('.')[0],folder, norm_size, printc=False)
        features = pd.concat([features, fts], ignore_index=True)

    return features

def create_all_features_full():
    all_feats = pd.read_csv('./all_features.csv', index_col=0)
    all_dist = pd.read_csv('./all_dist_feats.csv', index_col=0)
    all_dist.drop(columns=['n'],inplace=True)
    all_dist.sort_index(inplace=True)

    all_feats = pd.concat([all_feats.iloc[:,:3],all_dist,all_feats.iloc[:,3:]],axis=1)
    all_feats = all_feats.astype(float)
    all_feats.drop(columns=all_feats.columns[all_feats.std()==0],inplace=True)

    all_feats.drop(columns=['mst_dist_len', 'norm_dist_CV', 'dist_range', 'norm_dist_skew', 'norm_dist_kurt', 'norm_dist_lt_mean', 
                        'norm_dist_distinct', 'norm_dist_mode_freq', 'dist_centroid_mean', 'nn_dist_min', 'norm_dist_range', 
                        'mst_dists_mean', 'nn_dist_range', 'mst_dists_range', 'mst_degree_CV', 'mst_degree_range', 'cluster_centroid_dist_range'],inplace=True)
    # nans already replaced with 0

    # rename index to name
    all_feats.index.name = 'name'
    all_feats.to_csv('./all_features_full.csv')

if __name__ == '__main__':

    # # small instance
    # get_features0('A-n32-k5')    
    # # medium instance
    # get_features0('X-n200-k36')    
    # # large instance
    # get_features0('X-n856-k95')

    # fol = './Instances/'
    # get_features('XGen101_1232_01', f'{fol}VRP-X-gen', k=False, norm_size=1000, calc_k = True, printc = True)
    # get_features('Golden_14', '/home/noticed/PhD-VRP/Instances/VRP-instances0', k=False, norm_size=1000, calc_k = True, printc = True)
    
    # get_features_from_folder_p(2, '../Instances/Vrp-EGen/', k=False, simp_k=True, norm_size=1000, save='EGen_features.csv')
    # get_features_from_folder('../Instances', k=False, norm_size=1000, save='test_features.csv')

    get_dist_from_folder_p(2, '../Instances/ALL/', k=False, norm_size=1000, save='./all_dist_feats.csv')
    
    # small = CVRP.from_file('../Instances/', 'X-n200-k36', k_known=True)
    # small.normalise_instance(400)
    # small.plot_instance(norm=True)
    # small.plot_instance(norm=False)

    # scipy.spatial.distance.squareform(
    #     scipy.spatial.distance.pdist(small.norm_coords))