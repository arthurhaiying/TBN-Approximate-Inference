from collections import defaultdict
from pathlib import Path
import itertools as iter
import numpy as np
import sys

if __name__ == '__main__':
    basepath = Path(__file__).parents[1]
    sys.path.append(str(basepath))

from approximate.networks import NetParser
from approximate.sample import direct_sample_and_save_mp, split_data
from tbn.tbn import TBN
from tbn.node import Node
from tbn.testNode import TestNode
from tac import TAC
import train.data as Data
import utils.VE as VE

"""
    Build natural jointree for Bayesian Networks
"""

# return an adjacency list representation of bn
# dag: a dict that maps node name to list of children
def get_dag(bn):
    dag = {n.name:[] for n in bn.nodes} 
    for node in bn.nodes:
        x = node.name
        parents = [p.name for p in node.parents]
        for p in parents:
            dag[p].append(x)

    return dag

# return the underlying uag for dag
# uag: a dict that maps node name to list of neighbors
def get_uag(dag):
    uag = {x:[] for x in dag.keys()}
    for x,children in dag.items():
        for c in children:
            uag[x].append(c)
            uag[c].append(x)

    return uag

# return reversed dag
# dag2: a dict that maps node name to list of parents
def get_reversed_dag(dag):
    rev_dag = {x:[] for x in dag.keys()}
    for x,childen in dag.items():
        for c in childen:
            rev_dag[c].append(x)

    return rev_dag

# return a topological ordering of nodes in dag
def topo_order(dag):
    inDegrees = {x:0 for x in dag.keys()}
    for x,children in dag.items():
        for c in children:
            inDegrees[c]+=1

    queue = [x for x,degree in inDegrees.items() if degree == 0] # roots
    order = []
    while queue:
        x = queue.pop(0)
        order.append(x)
        for child in dag[x]:
            inDegrees[child]-=1 # decrement in-degrees for child
            if inDegrees[child] == 0:
                queue.append(child)

    return order

# Given a spanning tree T of G, convert T into polytree by 
# directing the edges in T with respect to G
def convert_to_polytree(dag, T):
    poly_T = {x:[] for x in T.keys()}
    for x,children in T.items():
        for c in children:
            if c in dag[x]:
                poly_T[x].append(c)
            else:
                poly_T[c].append(x)
    return poly_T



# return a spanning tree of dag
# T: a dict that maps node name to list of children 
def get_spanning_tree(dag):
    uag = get_uag(dag)
    inDegrees = {x:0 for x in dag.keys()}
    for x, chilren in dag.items():
        for c in chilren:
            inDegrees[c]+=1
    root = None
    for x,degree in inDegrees.items():
        if degree == 0:
            root = x
            break
    
    # run bfs from the first root
    queue = [root] 
    visited = set([root])
    T = {}
    while queue:
        x = queue.pop()
        T[x] = []
        for n in uag[x]: # neighbor
            if n not in visited:
                queue = [n] + queue
                visited.add(n)
                T[x].append(n)

    return T

# build a natural jointree (T, clusters) for bn
# T: a spanning tree for dag
# clusters: a dict that maps node in T to set of variables in dag
def make_natural_join_tree1(bn, trim=False): 
    # get spanning tree
    dag = get_dag(bn)
    T = get_spanning_tree(dag)
    hosts = {} # map cpts to jointree nodes
    for node in bn.nodes:
        x = node.name
        family = [n.name for n in node.family]
        hosts[x] = set(family)

    # get cluster assignments
    clusters = {x:set() for x in T.keys()}
    vars = defaultdict(dict) # vars[x][v] represent variables on the x-side of edge (x,v)
    seps = defaultdict(dict) # seps[x][v] represent separator on edge (x,v) 
    
    # first pass: visit nodes bottom up and set vars[x][parent]
    order = topo_order(T)
    T_rev = get_reversed_dag(T)
    for x in order[::-1]: # bottom up
        var = hosts[x].copy()
        children, parents = T[x], T_rev[x]
        for c in children:
            var |= vars[c][x]
        if len(parents) == 0: # root
            pass
        elif len(parents) == 1:
            p = parents[0]
            vars[x][p] = var
            #print("vars({}, {}) = {}".format(x,p,var))
        else:
            raise RuntimeError("T is not a tree!")

    # second pass: visit nodes top down and set vars[x][child]
    for x in order: 
        var = hosts[x].copy()
        children, parents = T[x], T_rev[x]
        for p in parents:
            var |= vars[p][x]
        for c in children:
            children2 = children.copy()
            children2.remove(c)
            vars[x][c] = var.union(*map(lambda n:vars[n][x], children2))
            #print("vars({}, {}) = {}".format(x,c,var))

    # set separators and clusters
    for x in order:
        for n in T[x]:
            sep = vars[x][n] & vars[n][x]
            seps[x][n] = sep
            seps[n][x] = sep

    for x in T.keys():
        cluster = hosts[x]
        children, parents = T[x], T_rev[x]
        for n in parents+children:
            cluster |= seps[n][x]
        clusters[x] = cluster

    # some variables can be trimmed from cluster of x if it is not prime varibale and
    # it does not appear in separator between x and its children
    if trim:
        for x, children in T.items():
            seps_below_x = [seps[x][c] for c in children]
            seps_below_x = set().union(*seps_below_x)
            cluster2 = []
            for n in clusters[x]:
                if n == x or n in seps_below_x:
                    cluster2.append(n)
            clusters[x] = cluster2

    return T, clusters


# build a natural join polytree (T, clusters) for bn
# T: a spanning tree for dag
# clusters: a dict that maps node in T to set of variables in dag
def make_natural_join_tree2(bn, trim=False): 
    # get spanning tree
    dag = get_dag(bn)
    T = get_spanning_tree(dag)
    hosts = {} # map cpts to jointree nodes
    for node in bn.nodes:
        x = node.name
        family = [n.name for n in node.family]
        hosts[x] = set(family)

    # get cluster assignments
    clusters = {x:set() for x in T.keys()}
    vars = defaultdict(dict) # vars[x][v] represent variables on the x-side of edge (x,v)
    seps = defaultdict(dict) # seps[x][v] represent separator on edge (x,v) 
    
    # first pass: visit nodes bottom up and set vars[x][parent]
    order = topo_order(T)
    T_rev = get_reversed_dag(T)
    for x in order[::-1]: # bottom up
        var = hosts[x].copy()
        children, parents = T[x], T_rev[x]
        for c in children:
            var |= vars[c][x]
        if len(parents) == 0: # root
            pass
        elif len(parents) == 1:
            p = parents[0]
            vars[x][p] = var
            #print("vars({}, {}) = {}".format(x,p,var))
        else:
            raise RuntimeError("T is not a tree!")

    # second pass: visit nodes top down and set vars[x][child]
    for x in order: 
        var = hosts[x].copy()
        children, parents = T[x], T_rev[x]
        for p in parents:
            var |= vars[p][x]
        for c in children:
            children2 = children.copy()
            children2.remove(c)
            vars[x][c] = var.union(*map(lambda n:vars[n][x], children2))
            #print("vars({}, {}) = {}".format(x,c,var))

    # set separators and clusters
    for x in order:
        for n in T[x]:
            sep = vars[x][n] & vars[n][x]
            seps[x][n] = sep
            seps[n][x] = sep

    for x in T.keys():
        cluster = hosts[x]
        children, parents = T[x], T_rev[x]
        for n in parents+children:
            cluster |= seps[n][x]
        clusters[x] = cluster

    # convert T into polytree
    T = convert_to_polytree(dag, T)

    # some variables can be trimmed from cluster of x if it is not prime varibale and
    # it does not appear in separator between x and its children
    if trim:
        for x, children in T.items():
            seps_below_x = [seps[x][c] for c in children]
            seps_below_x = set().union(*seps_below_x)
            cluster2 = []
            for n in clusters[x]:
                if n == x or n in seps_below_x:
                    cluster2.append(n)
            clusters[x] = cluster2

    return T, clusters




from graphviz import Digraph

def dot(dag, name="dot", node_labels=None, view=False):
    dot = Digraph()
    if node_labels is not None:
        for k,v in node_labels.items():
            dot.node(k, label=v)
    for x,children in dag.items():
        for c in children:
            dot.edge(x, c)
    dot.render('approximate/%s.gv'%name, view=view)

def test_join_tree():
    net_filepath = "approximate/networks/asia2.net"
    bn = NetParser.parseBN(net_filepath)
    G = get_dag(bn)
    dot(G, name="true", view=True)
    T, clusters = make_natural_join_tree2(bn, trim=True)
    print("tree: {}".format(T) )
    dot(T, name="tree", view=True)
    print("cluster: {}".format(clusters))
    node_labels = {n: "".join(cls) for n,cls in clusters.items()}
    dot(T, name="jointree", node_labels=node_labels, view=True)

# Return CPT with reduced parents by taking average CPT over consistent parent instantations
# cpt (np array): original cpt over parents1
# parents2: must be subset of parent1
# return: reduced CPT over parents2 
def make_reduced_cpt(cpt, parents1, parents2):
    assert set(parents2).issubset(parents1)
    sparents, tparents = [], []
    for p in parents1:
        if p in parents2:
            tparents.append(p)
        else:
            sparents.append(p)
    sum_axis = tuple(parents1.index(p) for p in sparents)
    cpt2 = np.mean(cpt, axis=sum_axis)
    transpose_axis = tuple(tparents.index(p) for p in parents2)
    transpose_axis += (-1,)
    cpt2 = np.transpose(cpt2, axes=transpose_axis)
    return cpt2


def make_incomplete_bn(bn, T):
    bn2 = TBN("incomplete bn")
    name2node = {}
    T_rev = get_reversed_dag(T)
    order = topo_order(T)
    for x in order:  # topological order
        node = bn.node(x)
        values, parents1, cpt = node.values, node.parents, node.tabular_cpt()
        parents2 = T_rev[x]
        parents2 = [name2node[p] for p in parents2]
        pnames1 = [p.name for p in parents1]
        pnames2 = [p.name for p in parents2]
        cpt2 = make_reduced_cpt(cpt, pnames1, pnames2)
        node2 = Node(x, values=values, parents=parents2, cpt=cpt2)
        name2node[x] = node2
        bn2.add(node2)

    return bn2


# Given a bn and a query (evidence, Q) and a jointree T over cluster states 
# find a set of nodes and selection evidence so that if we inject testing on these nodes
# we can fully recover G using T over simple states
# return: dict(node -> list of nodes) that represents selection evidence for each testing node 
# naive: make all non-root nodes testing 
def make_testing_scheme1(bn, T, evidence):
    # TODO: testing order
    nodes = T.keys()
    inDegrees = {x:0 for x in T.keys()}
    for x, chilren in T.items():
        for c in chilren:
            inDegrees[c]+=1
    roots = [x for x,degree in inDegrees.items() if degree == 0]
    testing = set(nodes) - set(roots)
    sel_evidence_dict = {x:evidence for x in testing}
    return sel_evidence_dict

# naive: make all nodes testing
def make_testing_scheme2(bn, T, evidence):
    testing = T.keys()
    sel_evidence_dict = {x:evidence for x in testing}
    return sel_evidence_dict




def make_testing_scheme1(bn, T, evidence):
    # naive: make every non-root nodes testing 
    # TODO: testing order
    nodes = T.keys()
    inDegrees = {x:0 for x in T.keys()}
    for x, chilren in T.items():
        for c in chilren:
            inDegrees[c]+=1
    roots = [x for x,degree in inDegrees.items() if degree == 0]
    testing = set(nodes) - set(roots)
    sel_evidence_dict = {x:evidence for x in testing}
    return sel_evidence_dict


def make_incomplete_tbn(bn, T, clusters, evidence, Q, testing_type='testing', num_cpts=None):
    testing_types = ('testing', 'testing_by_evd')
    if testing_type not in testing_types:
        raise ValueError(f"testing type {testing_type} is not valid")
    if testing_type == 'testing_by_evd' and num_cpts is None:
        raise ValueError(f"num_cpts should be provided for testing_by_evd")

    tbn = TBN("incomplete tbn")
    name2node = {}
    # choose testing scheme
    sel_evidence_dict = make_testing_scheme1(bn, T, evidence)
    testing = sel_evidence_dict.keys()
    print("testing: ", testing)

    # make vanilla tbn
    if testing_type == 'testing':
        order = topo_order(T)
        T_rev = get_reversed_dag(T)
        for x in order: 
            node = bn.node(x)
            values, parents1, cpt = node.values, node.parents, node.tabular_cpt()
            parents2 = T_rev[x]
            parents2 = [name2node[p] for p in parents2]
            pnames1 = [p.name for p in parents1]
            pnames2 = [p.name for p in parents2]
            cpt2 = make_reduced_cpt(cpt, pnames1, pnames2)
            if x in testing:
                try:
                    node2 = Node(x, values=values, parents=parents2, testing=True, cpt=cpt2)
                except:
                    print(f"Testing Node {x} has parents {parents2}")
                    exit(1)
            else:
                node2 = Node(x, values=values, parents=parents2, cpt=cpt2)
            name2node[x] = node2
            tbn.add(node2)

    else:  # make tbn that test by evidence
        order = topo_order(T)
        T_rev = get_reversed_dag(T)
        for x in order: 
            node = bn.node(x)
            values, parents1, cpt = node.values, node.parents, node.tabular_cpt()
            parents2 = T_rev[x]
            parents2 = [name2node[p] for p in parents2]
            pnames1 = [p.name for p in parents1]
            pnames2 = [p.name for p in parents2]
            cpt2 = make_reduced_cpt(cpt, pnames1, pnames2)
            if x in testing:
                node2 = TestNode(x, values=values, parents=parents2, num_cpts=num_cpts, cpt=cpt2)
            else:
                node2 = Node(x, values=values, parents=parents2, cpt=cpt2)
            name2node[x] = node2
            tbn.add(node2)

        # for testing nodes, set selection evidences 
        for x,evidence in sel_evidence_dict.items():
            node = name2node[x]
            evidence = [name2node[e] for e in evidence]
            node.set_selection_evidence(evidence)

    return tbn

# clip small values in distribution
def clip(dist):
    EPSILON = np.finfo('float32').eps
    safe_dist = np.where(np.less(dist, EPSILON), EPSILON, dist)
    return safe_dist

# computes Kullback-Leibler divergence score between dist_true and dist_pred
def KL_divergence(dist_true, dist_pred):
    assert dist_true.shape == dist_pred.shape
    batch_size = dist_true.shape[0]
    dist_true = clip(dist_true).reshape((batch_size,-1)) # clip and flatten
    dist_pred = clip(dist_pred).reshape((batch_size,-1)) 
    kl_loss = dist_true * np.log(dist_true/dist_pred)
    kl_loss = np.sum(kl_loss, axis=-1)
    kl_loss = np.mean(kl_loss)
    return kl_loss


def evaluate(bn, evidence, Q, tac_list):
    cards = [len(bn.node(e).values) for e in evidence]
    inst = list(iter.product(*map(range, cards)))
    evidence_col = Data.evd_id2col(inst, cards)
    marginal = VE.posteriors(bn, evidence, Q, evidence_col) # true marginal
    loss_list = []
    for tac in tac_list:
        marginal_pred = tac.evaluate(evidence_col) # predicted marginal
        loss = KL_divergence(marginal, marginal_pred)
        loss_list.append(loss)

    return loss_list


def main(sample_size):
    net_filepath = "approximate/networks/barley.net"
    data_filename = "barley.csv"
    bn = NetParser.parseBN(net_filepath)
    dag = get_dag(bn)
    Q = "protein"
    evidence = [x for x,children in dag.items() if len(children)==0 and x != Q]
    ecards = [len(bn.node(e).values) for e in evidence]
    qcard = len(bn.node(Q).values)
    print(f"evidence: {evidence} Q: {Q}")
    # step 1: build incomplete bn / tbn 
    T, clusters = make_natural_join_tree2(bn, trim=True)
    dot(T, name="incomplete barley", view=True)
    incomplete_bn = make_incomplete_bn(bn, T)
    incomplete_tbn1 = make_incomplete_tbn(bn, T, clusters, evidence, Q, testing_type='testing')
    #incomplete_tbn2 = make_incomplete_tbn(bn, T, clusters, evidence, Q, testing_type='testing_by_evd',num_cpts=1)
    incomplete_tbn3 = make_incomplete_tbn(bn, T, clusters, evidence, Q, testing_type='testing_by_evd',num_cpts=3)
    # step 2: compile incomplete ac/tac from incomplete bn / tbn
    ac = TAC(incomplete_bn, inputs=evidence, output=Q, trainable=True)
    tac1 = TAC(incomplete_tbn1, inputs=evidence, output=Q, trainable=True)
    #tac2 = TAC(incomplete_tbn2, inputs=evidence, output=Q, trainable=True)
    tac3 = TAC(incomplete_tbn3, inputs=evidence, output=Q, trainable=True)
    # step 3: sample data from true bn
    data, order = direct_sample_and_save_mp(bn, sample_size)
    evidences, marginals = split_data(data, order, evidence, Q)
    evidences = Data.evd_id2col(evidences, ecards)
    marginals = Data.mar_id2mar(marginals, qcard)
    # step 4: fit incomplete ac / tac
    ac.fit(evidences, marginals, loss_type='CE',metric_type='CE')
    tac1.fit(evidences, marginals, loss_type='CE',metric_type='CE')
    #tac2.fit(evidences, marginals, loss_type='CE',metric_type='CE')
    tac3.fit(evidences, marginals, loss_type='CE',metric_type='CE')
    tac_types = ['ac', 'tac1','tac3']
    tac_list = [ac, tac1,tac3]
    loss_list = evaluate(bn, evidence, Q, tac_list)
    np.set_printoptions(precision=3)
    print("KL loss ------------------------ ")
    for type,loss in zip(tac_types, loss_list):
        print("{}: {}".format(type, loss))

    




    
if __name__ == '__main__':
    #test_join_tree()
    main(5000)

        

        

