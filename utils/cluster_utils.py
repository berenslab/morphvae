from sklearn.cluster import k_means
from .graph_utils import make_directed_tree
import numpy as np
import torch

def _convert_cluster_results_dict_into_array(shape, results_dict):
    clustered_rws = np.ones(shape)*-np.infty
    
    for k in results_dict.keys():
        rw_indices = results_dict[k]['indices']
        labels = results_dict[k]['labels']
        means_index = labels - labels.min()
        clustered_rws[rw_indices,k] = results_dict[k]['means'][means_index]
    return clustered_rws
def _check_for_cycles(results_dict):
    
    has_cycle = False
    n_cycles = 0
    l = len(results_dict)
    keys = list(results_dict.keys())
    if l > 3:
        index = np.array([list(results_dict[keys[-2]]['indices']).index(i) 
                          for i in results_dict[keys[-1]]['indices']])
        temp = [results_dict[keys[-1]]['labels'].reshape(-1,1),
                results_dict[keys[-2]]['labels'][index].reshape(-1,1)]
        unique_paths = np.unique(np.concatenate(temp, axis=1), axis=0)
        _, counts = np.unique(unique_paths[:,0], return_counts=True)
        has_cycle = (counts > 1).any()
        n_cycles = (counts >1).sum()
    return has_cycle, n_cycles

def get_no_intersections(rw_rep):
    walk_length = rw_rep.shape[1]
    no_intersections = []
    for j in range(walk_length):
        u = np.unique(rw_rep[:,j,:], axis=0)
        if -np.infty in u:
            u = u[1:]
        if len(u) > 0:
            no_intersections.append(len(u))
        else:
            break
    return no_intersections

def get_clustered_rws_kmeans(sampled_rws, no_intersections, verbose=False):
    
    si = sampled_rws
    steps = len(no_intersections) -1
    cluster_result = {}
    
    # to label the clusters consecutively
    cluster_label_offset = 0
    for k in range(steps,-1,-1):

        p = si[:,k,:]    
        p_index = p[:,0] != -np.infty
        p = p[p_index]
        
        if len(p) > 0:
            no_clus = min(no_intersections[k], len(p))
            if verbose: 
                print("Clustering step %i with %i clusters..."%(k,no_clus))
            cluster_means, labels, _= k_means(p,n_clusters=no_clus, random_state=42)
            cluster_result[k] = dict(means=cluster_means, 
                                    labels=(labels + cluster_label_offset), 
                                    indices = np.where(p_index)[0] )
                                
            cluster_label_offset += len(cluster_means)    
            
    ## get the clustered random walks
    clustered_rws = _convert_cluster_results_dict_into_array(si.shape, cluster_result)
    return cluster_result, clustered_rws

from sklearn.cluster import AgglomerativeClustering
def get_clustered_rws_agglom(sampled_rws, no_intersections = None, dist_thresh=.5):
    
    si = sampled_rws
    steps = si.shape[1]-1
    cluster_result = {}
    last_cluster = {}
    # to label the clusters consecutively
    cluster_label_offset = 0
    for k in range(steps,-1,-1):

        p = si[:,k,:]    
        p_index = p[:,0] != -np.infty
        p = p[p_index]
        label_dict = {}
        walk_indices = np.where(p_index)[0]
        if len(p) > 0:
            labels = np.array([0]).astype(int)
            cluster_means = p
            if len(p) > 1:
                
                if no_intersections is not None:
                    no_clus = min(no_intersections[k], len(p))
                    algo = AgglomerativeClustering(n_clusters=no_clus)
                else:
                    algo = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh)
                
                algo.fit(p)
                labels = algo.labels_
                    
#                 # check for a split
                if len(last_cluster) > 0:
                    for key in last_cluster[k+1].keys():
                        # get the indices of each cluster of the step before
                        key_values = last_cluster[k+1][key]        
                        w_index = np.array([np.where(walk_indices == kv)[0][0] for kv in key_values])

                        split = np.unique(labels[w_index])
                        if len(split) > 1:
                            ## A split happended!

                            # merge the clusters
                            # create the index
                            label_index = np.array([False]*len(labels))
                            for s in split:
                                label_index = label_index | (labels == s)

                            labels[label_index] = split[0]
                # rename the labels to go from 0 to num clusters
                unique_labels = np.unique(labels)
                new_labels = np.zeros_like(labels)
                for i, l in enumerate(unique_labels):
                    new_labels[labels == l] = i
                labels = new_labels
                    
                no_clus = len(unique_labels)
                cluster_means = np.zeros((no_clus, 3))
                for l in range(no_clus):
                    label_index = labels == l
                    cluster_means[l] = p[label_index].mean(dim=0)
                    label_dict[l] = list(walk_indices[label_index])

    
            cluster_result[k] = dict(means=cluster_means, 
                                    labels=(labels + cluster_label_offset), 
                                    indices = walk_indices )
                                
            cluster_label_offset += len(cluster_means)    
        
        last_cluster[k] = label_dict
            
    ## get the clustered random walks

    clustered_rws = _convert_cluster_results_dict_into_array(si.shape, cluster_result)
    return cluster_result, clustered_rws

def relabel_cluster_ids(cluster_result):
    
    offset = 0
    for i, item in cluster_result.items():
        item['labels'] += offset
        offset += len(np.unique(item['labels']))
    

def cluster_rws_soma2tip(sampled_rws, n_intersections, dist_thresh=.5):
    
    steps = sampled_rws.shape[1]-1
    cluster_result = {}
    
    for k in range(0,steps):
        
        # p contains the points to be clustered
        p = sampled_rws[:,k,:]    
        p_index = p[:,0] != -np.infty
        p = p[p_index]
        walk_indices = np.where(p_index)[0]
        
        if len(p) > 0:
            labels = np.array([0]).astype(int)
            if len(p) > 1:
                # cluster p
                if k < len(n_intersections):
                    c = min((n_intersections[k], len(p)))
                    algo = AgglomerativeClustering(n_clusters=c)
                else:    
                    algo = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh)
                
                algo.fit(p)
                labels = algo.labels_
                
                if k > 0:
                    # has a merge occured? Then split
                    labels = split_cluster(labels, walk_indices, cluster_result[k-1])
                
                no_clus = len(np.unique(labels))
                cluster_means = np.zeros((no_clus, 3))
                for l in range(no_clus):
                    label_index = labels == l
                    cluster_means[l] = p[label_index].mean(dim=0)
                    
                cluster_result[k] = dict(means=cluster_means, 
                                    labels=labels, 
                                    indices = walk_indices)
    relabel_cluster_ids(cluster_result)
    return cluster_result

def split_cluster(labels, indices, prev_result):
    """
        Reassigns new labels for samples which had different parents in the previous cluster step.
        Makes sure that walks are not merged along the way to the tips.
        returns labels (list of int)
    """

    cluster = np.unique(labels)
    for clus in cluster:
        # get parent clusters
        label_index = labels == clus
        walk_indices = indices[label_index]

        label_index_prev_step = [np.where(prev_result['indices']==i)[0][0] for i in walk_indices]
        parent_cluster = prev_result['labels'][label_index_prev_step]

        unique_parents = np.unique(parent_cluster)
        if len(unique_parents) > 1:
            # split cluster
            for p in unique_parents[1:]:

                new_label = labels.max() + 1
                parent_index = parent_cluster == p
                relabel_index = np.where(label_index)[0][parent_index]
                labels[relabel_index] = new_label
            
    return labels


import networkx as nx
from morphopy.neurontree import NeuronTree as nt
# reconstruct the trees
def tree_from_clustered_result(cluster_result, n_walks=256):
    soma = cluster_result[0]['labels'][0]
    means = np.concatenate([c['means'] for k,c in cluster_result.items()])
    walk_length = np.array(list(cluster_result.keys())).max() + 1
    rw_in_labels = np.ones((n_walks, walk_length)).astype(int)*-np.infty

    for k, item in cluster_result.items():

        rw_in_labels[item['indices'],k] = item['labels'] 

    # only take paths that are longer than 1 non padded coordinates
    rw_in_labels = rw_in_labels[(rw_in_labels != -np.infty).sum(axis=1) > 1]
    flattened_rw_in_labels = rw_in_labels[rw_in_labels != -np.infty].astype(int)

    # # get nodes
    nodes = [(n, {'pos':means[n], 'type':3, 'radius':1}) for n in np.unique(flattened_rw_in_labels)]
    edges = [z for z in list(zip(flattened_rw_in_labels, flattened_rw_in_labels[1:])) if z[1] !=soma]

    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)
    G.nodes[soma]['type'] = 1

    N = nt.NeuronTree(graph=G)
    return N



