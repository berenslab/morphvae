import numpy as np
import networkx as nx


from morphopy.neurontree.utils import eulerAnglesToRotationMatrix
from morphopy.neurontree import NeuronTree as nt

from .graph_utils import get_barcode, draw_tree, make_directed_tree

## plotting ###

import matplotlib.pyplot as plt
import seaborn as sns

class ToyNeuron:
    
    def __init__(self, dim=3, min_degree=-20, max_degree=20, lam=2):
        
        assert dim <=3
        assert min_degree < max_degree
        
        self.dim = dim
        self.lam=lam
        self.node_dict = {0: {'pos': np.array([0]*dim)}}
        self.edge_list = []
        self.min_degree = min_degree
        self.max_degree = max_degree
    
    def rotate(self, start):  
        # sample angle # TO DO make this an exponentially decaying function
        theta = np.random.randint(low=self.min_degree, high= self.max_degree)*np.pi/180
        if self.dim ==2: 
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        elif self.dim == 3:
            R = eulerAnglesToRotationMatrix([0, 0, theta])
        return R@start

    def grow(self, start_pos, direction, noise):
        
        return start_pos + self.rotate(direction) + noise

    def grow_branch(self, V=50, direction=1):
        """
        V int: number on nodes to grow
        direction int: direction to grow into. 1: up [0,1,0], 2: right [1,0,0], 3: down [0,-1,0], 4: left [-1,0,0]
        """
        # v holds the number of nodes that have been created already
        v = 0
        
        # defines in wich direction the branch is growing
        if direction == 1:
            vec = np.array([0,1,0])
        elif direction == 2 :
            vec = np.array([1,0,0])
        elif direction == 3 :
            vec = np.array([0,-1,0])
        elif direction == 4 :
            vec = np.array([-1,0,0])
            
        node_ids = [0]
        while len(node_ids) > 0 and v < V:

            start_id = node_ids.pop(0)
            max_id = max(self.node_dict.keys())
            
            # sample number of branches 
            branch_order = np.max((1, np.random.poisson(lam=self.lam)))

            for branch in range(1, branch_order+1):
                
                if v < V: 
                    current_node_id =  max_id + branch

                    # sample new position
                    start_pos = self.node_dict[start_id]['pos']
                    pos = self.grow(start_pos, vec, np.random.normal(loc=0, scale=.01, size=self.dim))
                    self.node_dict[current_node_id] = {'pos': pos}
                    
                  
                    # compute eucildean dist
                    ec = np.linalg.norm(pos - start_pos)
                    # add edge
                  
                    self.edge_list += [(start_id, current_node_id, dict(euclidean_dist=ec, path_length=ec))]
                    node_ids += [current_node_id]
                    
                    v +=1

    def grow_neuron(self, V=100, directions=None):
        
        if directions is None:
            directions=[1,3]
        
        splits = len(directions)
            
        v = np.floor(V/splits)

        no_nodes = [v]*(splits-1) + [V- v*(splits-1)]

        for d, nodes in zip(directions, no_nodes):
            self.grow_branch(direction=d, V=nodes)
                
    def rescale(self, height=1, width=1, depth=1):
        
        old_positions = np.array([d['pos'] for k, d in self.node_dict.items()])
        min_ = np.min(old_positions, axis=0)
        max_ = np.max(old_positions, axis=0)
        extend = max_ - min_
        V = len(self.node_dict)
        
        new_positions = (old_positions-min_)/extend
        new_positions *= np.array([width, height, depth])
        
        new_positions -= new_positions[0,:]
        new_nodes = {k: {'pos':new_positions[k]} for k in range(V)}
        
        self.node_dict.update(new_nodes)
        
        
    def make_neurontree(self):
        G = nx.DiGraph()
        for n, attr in self.node_dict.items():
            G.add_node(n, pos=attr['pos'], type=3, radius=1)
        G.add_edges_from(self.edge_list)

        neuron = nt.NeuronTree(graph=G)
        return neuron
    

    
def plot_population_sample(population_parameter, n_samples=5):
    
    colors = sns.color_palette('pink', n_colors=len(population_parameter)+1)
    fig = plt.figure(figsize=(n_samples*len(population_parameter),3))
    offset=0
    
    for f, params in enumerate(population_parameter):
        
        try:
            directions = params['directions']
        except KeyError:
            directions=None
        try:
            min_degree = params['min_degree']
        except KeyError:
            min_degree = -50
            
        try:
            max_degree = params['max_degree']
        except KeyError:
            max_degree = 150
        
        for k in range(n_samples):

            tn = ToyNeuron(lam=params['lam'], min_degree=min_degree,max_degree=max_degree)

            tn.grow_neuron(V=params['V'], directions=directions)
            rescaling_height= np.max([1, params['height']])
            rescaling_width = np.max([1,params['width']])
            rescaling_depth = np.max([1,params['depth']])
            tn.rescale(height=rescaling_height, width=rescaling_width, depth=rescaling_depth)
            neuron = tn.make_neurontree()
            c = colors[f]
            neuron.draw_2D(ax=plt.gca(), projection='xy', x_offset = offset, dendrite_color=c)
            plt.scatter(offset, 0, c='k', marker='s', zorder=10, s=8)
            offset += (np.max(neuron.get_extent()) + 3)
    return fig

from collections import OrderedDict
def generate_toy_data(params_list, with_pos=False):
    codes_asc = []
    codes_desc = []
    neurons = []

    for params in params_list:
        
        N = params['N']
        
        
        try:
            mean_height = params['height']
        except KeyError:
            mean_height = 1
            
        try:
            mean_width = params['width']
        except KeyError:
            mean_width = 1
        
        try:
            mean_depth = params['depth']
        except KeyError:
            mean_depth = 1
            
        try:
            directions = params['directions']
        except KeyError:
            directions=None
            
        try:
            min_degree = params['min_degree']
        except KeyError:
            min_degree = -50
            
        try:
            max_degree = params['max_degree']
        except KeyError:
            max_degree = 150   
            
        
        heights = np.random.normal(loc=mean_height, scale=.2, size=N)
        widths = np.random.normal(loc=mean_width, scale=.2, size=N)
        depths = np.random.normal(loc=mean_depth, scale=.2, size=N)
        for k in range(N):

            r = 0
            tn = ToyNeuron(lam=params['lam'], min_degree=min_degree,max_degree=max_degree)

            tn.grow_neuron(V=params['V'], directions=directions)
            tn.rescale(height=heights[k], width=widths[k], depth=depths[k])
            neuron = tn.make_neurontree()

            bc_asc, order_asc = get_barcode(neuron.get_graph(), root=r)
            bc_desc, order_desc = get_barcode(neuron.get_graph(), root=r, order='desc')

            if with_pos:
                # get positions --> N X 3
                positions_dict = neuron.get_node_attributes('pos')
                sorted_pos_dict = OrderedDict(sorted(positions_dict.items()))
                positions_asc = np.array(list(sorted_pos_dict.values()))[order_asc,:]
                codes_asc.append(np.hstack((bc_asc.reshape(-1,1),positions_asc)))
                
                positions_desc = np.array(list(sorted_pos_dict.values()))[order_desc,:]
                codes_desc.append(np.hstack((bc_desc.reshape(-1,1),positions_desc)))                              
                
            else:
                codes_asc.append(bc_asc)
                codes_desc.append(bc_desc)
            neurons.append(neuron)
    return codes_asc, codes_desc, neurons