### UTILITY FUNCTIONS FOR WORKING WITH GRAPHS AND NEURON_GRAPHS


from copy import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_cumulative_length(G, root=0, weight=None):
    # get leaves
    out_degrees = dict(G.out_degree())
    nodes = np.array(list(out_degrees.keys()))
    degree_values = np.array(list(out_degrees.values()))
    leaves = nodes[degree_values == 0]

    # sort leaves by length to soma descending
    pl_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))[root]
    leaf_order = np.argsort([pl_dict[l] for l in leaves])[::-1]
    active_nodes = list(leaves[leaf_order])

    nodes = G.nodes()
    c_pl = dict(zip(nodes, [0] * len(nodes)))

    while len(active_nodes) > 0:
        a = active_nodes.pop(0)

        edges = G.edges(a, data=True)
        # for add pl of each edge coming from a
        for e1, e2, data in edges:
            if weight is None:
                c_pl[e1] += (c_pl[e2] + 1)
            else:
                c_pl[e1] += (c_pl[e2] + data[weight])
        # insert the parents
        parent = list(G.predecessors(a))
        if parent:  # in case a is the root and the parent list is empty
            if parent[0] not in active_nodes:
                active_nodes += parent

    return c_pl


def get_sorted_node_ids(T, root=0, weight=None, order='asc'):
    node_ids = np.array(list(T.nodes()))
    root_ix = np.where(node_ids == root)[0][0]

    pl_from_root = get_cumulative_length(T, root, weight)

    active_nodes = []
    active_nodes.append(root)

    # fill sorted_nodes list
    sorted_nodes = []
    while len(active_nodes) > 0:

        n = active_nodes.pop(0)
        sorted_nodes += [n]
        successors = list(T.successors(n))
        if successors:
            pl = [pl_from_root[s] for s in successors]
            s_idx = np.argsort(pl)
            if order == 'desc':
                s_idx = s_idx[::-1]
            active_nodes = [successors[s] for s in s_idx] + active_nodes
    return sorted_nodes


def get_barcode(T, root=0, weight=None, order='asc'):
    # get the topological order of the tree
    top_order = get_sorted_node_ids(T, root=root, weight=weight, order=order)
    node_ids = np.array(list(T.nodes()))
    index_order = [np.where(node_ids == t)[0][0] for t in top_order]
    bct = np.array(np.sum(nx.adj_matrix(T)[:, index_order][index_order, :].T, axis=0))[0]
    return bct, index_order


def separate_words(bct):
    new_string = ''
    tokens = copy(bct.split(" "))
    token = tokens.pop(0)
    while len(tokens) > 0:

        next_token = tokens.pop(0)
        new_string += token

        n_zeros = 0
        while (next_token == '0'):  # count trailing zeros if there are any
            new_string += next_token
            n_zeros += 1
            if len(tokens) > 0:
                next_token = tokens.pop(0)
            else:
                break

        if n_zeros > 0:  # if there has been zeros --> C or T word
            new_string += " "
            token = next_token

        else:
            if int(token) > 1:  # --> we have an arborization
                new_string += " "
                token = next_token
            else:
                n_ones = 0
                while (next_token == '1'):
                    n_ones += 1
                    new_string += next_token

                    if len(tokens) > 0:
                        next_token = tokens.pop(0)
                    else:
                        break

                if next_token == '0':  # we have to add the 10 word
                    # adjust string
                    # new_string = new_string[:-1]
                    # new_string += ' 10 '
                    new_string += next_token
                    new_string += " "
                    if len(tokens) > 0:
                        token = tokens.pop(0)
                else:  # just add a space
                    new_string += " "
                    token = next_token

    return new_string.replace("  ", " ")


def draw_tree(T):
    plt.figure()
    pos = nx.spring_layout(T)
    nx.draw(T, pos)
    _ = nx.draw_networkx_labels(T, pos, font_color='white')


def make_directed_tree(T, root=0):
    out_edges = list(T.out_edges())
    kept_edges = []
    successors = [root]
    visited = []

    while len(successors) > 0:

        visited += [successors.pop(0)]
        s = visited[-1]

        to_keep = [e for e in out_edges if e[0] == s]
        for e in to_keep:
            out_edges.remove((e[1], e[0]))

        potential_successors = list(nx.DiGraph.successors(T, s))
        for v in visited:
            if v in potential_successors:
                potential_successors.remove(v)

        successors += potential_successors

    return nx.from_edgelist(out_edges, create_using=nx.DiGraph)