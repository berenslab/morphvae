import numpy as np
import networkx as nx
import pandas as pd
import random

from morphopy.neurontree import NeuronTree as nt
from shapely.geometry import MultiLineString, Point, LineString

def get_geometry(neuron):
    sorted_keys = list(neuron.nodes())
    sorted_keys.sort()
    geometry = np.array([neuron.get_node_attributes('pos')[key] for key in sorted_keys])
    return geometry, np.array(sorted_keys)

def sample_random_walk(A, walk_length=50,start_idx=0, P=None):
    
    random_walk_indices = np.ones(walk_length)*np.infty*-1
    random_walk_indices[0] = start_idx
    idx = start_idx
    p = None
    for k in range(1,walk_length):
        
        
        neighbours = A[idx,:].indices
        # no jumping backwards
        neighbours = list(set(neighbours) - set(random_walk_indices[:k-1]))
        
        if len(neighbours) > 0:
            neighbours = np.array(neighbours)
            if P is not None:
                p = P[idx,neighbours] + 1/len(neighbours)
                p /= p.sum()
            idx = np.random.choice(neighbours, p=p)
            random_walk_indices[k] = idx
        else:
            break
    return random_walk_indices
    
def get_rw_representation(neuron, walk_length=16, n_walks=256, weighted=False):
    
    G = neuron.get_graph().to_undirected() #important make the tree undirected otherwise we cannot escape the leaves
    geometry, sorted_keys = get_geometry(neuron)
    
    A = nx.adjacency_matrix(G, nodelist=sorted_keys, weight='path_length')
    V = A.shape[0]
    
    if weighted:
        cumulative_path_length = neuron.get_cumulative_path_length()
        cpl = np.diag([cumulative_path_length[n] for n in sorted_keys])
        cpl = ((A != 0)@cpl)
        summed_per_row = cpl.sum(axis=1).reshape(-1,1)
        summed_per_row[summed_per_row == 0] = .1
        P = cpl / summed_per_row
    else:
        P = None
    
    random_walk_coordinates = np.ones((n_walks,walk_length,3)) *np.infty*-1     # holds the 3D coordinates of the visited nodes
    random_walk_indices = np.ones((n_walks,walk_length))*np.infty*-1 # holds the visited node ids

    # sample random walks
    for n in range(n_walks):
        
        sampled_random_walk = sample_random_walk(A,walk_length=walk_length,start_idx=0, P=P)
        sampled_random_walk = sampled_random_walk[sampled_random_walk != np.infty*-1]
        random_walk_indices[n, :len(sampled_random_walk)] = sampled_random_walk
        random_walk_coordinates[n, :len(sampled_random_walk)] = geometry[sampled_random_walk.astype(int)]
        
    return random_walk_coordinates, random_walk_indices

def get_possible_paths(neuron):
    
    tips = neuron.get_tips()
    sp = nx.shortest_path(neuron.get_graph(), source=neuron.get_root())

    geometry, sorted_nodes = get_geometry(neuron)
    possible_paths = [geometry[[np.where(sorted_nodes == n)[0][0] for n in sp[t]]] for t in tips]
    
    return possible_paths
    
def get_walk_representation(neuron, walk_length=16, n_walks=256):
    
    possible_paths = get_possible_paths(neuron)
    chosen_paths = [random.choice(possible_paths) for _ in range(n_walks)]

    walks = np.ones((n_walks,walk_length,3)) *np.infty*-1     # holds the 3D coordinates of the visited nodes
    for k, p in enumerate(chosen_paths):
        l = len(p)
        if l <= walk_length:
            walks[k,:l] = p
        else:
            walks[k,:] = p[:walk_length]
    return walks 
    
import os
def load_neurons(path, sort=True):
    neurons = []
    for path, _, files in os.walk(path):
    
        if sort:
            file_idx = np.argsort([int(f[:-4]) for f in files])
            files = np.array(files)[file_idx]
   
        for f in files:
            swc = pd.read_csv(path+f, delim_whitespace=True, comment='#',
                      names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False)
            
            N = nt.NeuronTree(swc=swc)
            neurons.append(N)
    return neurons


def get_sholl_intersection_profile(self, proj='xy', n_steps=36, centroid='centroid', sampling='uniform'):
    """
    Calculates the Sholl intersection profile of the neurons projection determined by the parameter _proj_. The
    Sholl intersection profile counts the intersection of the neurites with concentric circles with increasing
    radii. The origin of the concentric circles can be chosen to either be the centroid of the
    projected neuron's convex hull or to be the soma.

    :param proj: 2D projection of all neurites. Options are 'xy', 'xz' and 'yz'
    :param steps: number of concentric circles centered around _centroid_. Their radii are determined as the
     respective fraction of the distance from the centroid to the farthest point of the convex hull.
    :param centroid: Determines the origin of the concentric circles. Options are 'centroid' or 'soma'.
    :return: intersections  list, len(steps), count of intersections with each circle.
     intervals list, len(steps +1), radii of the concentric circles.
    """

    G = self.get_graph()
    coordinates = []

    if proj == 'xy':
        indx = [0, 1]
    elif proj == 'xz':
        indx = [0, 2]
    elif proj == 'yz':
        indx = [1, 2]
    else:
        raise ValueError("Projection %s not implemented" % proj)

    # get the coordinates of the points
    for e in G.edges():
        if self._nxversion == 2:
            # changed for version 2.x of networkX
            p1 = np.round(G.nodes[e[0]]['pos'], 2)
            p2 = np.round(G.nodes[e[1]]['pos'], 2)
        else:
            p1 = np.round(G.node[e[0]]['pos'], 2)
            p2 = np.round(G.node[e[1]]['pos'], 2)
        coordinates.append((p1[indx], p2[indx]))

    # remove illegal points
    coords = [c for c in coordinates if (c[0][0] != c[1][0] or c[0][1] != c[1][1])]

    lines = MultiLineString(coords).buffer(0.0001)
    bounds = np.array(lines.bounds).reshape(2, 2).T
    if centroid == 'centroid':
        center = np.array(lines.convex_hull.centroid.coords[0])
        p_circle = Point(center)
    elif centroid == 'soma':
        if self._nxversion == 2:
            # changed for version 2.x of networkX
            center = G.nodes[self.get_root()]['pos'][indx]
        else:
            center = G.node[self.get_root()]['pos'][indx]
        p_circle = Point(center)
    else:
        raise ValueError("Centroid %s is not defined" % centroid)

    # get the maximal absolute coordinates in x and y
    idx = np.argmax(np.abs(bounds), axis=1)
    r_max = np.linalg.norm(center - bounds[idx, [0, 1]])
    
    # get the Sholl intersection steps
    if sampling == 'log':
        steps = np.logspace(np.log(0.001),stop=np.log(r_max),num=n_steps, base=np.e)
    else:
        steps = np.linspace(0,r_max, n_steps)

    intersections = []
    intervals = [0]
    for r in steps[1:]:

        c = p_circle.buffer(r).boundary

        i = c.intersection(lines)
        if type(i) in [Point, LineString]:
            intersections.append(1)
        else:
            intersections.append(len(i))
        intervals.append(r)

    return intersections, intervals
