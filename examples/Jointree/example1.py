from pathlib import Path
import sys

if __name__ == '__main__':
    libpath = Path(__file__).resolve().parents[2]
    print("libpath: {}".format(libpath))
    sys.path.append(str(libpath))

from PyTAC.tbn.tbn import TBN
from PyTAC.tbn.node import Node
from PyTAC.utils import VE


import numpy as np
import random

# returns a random cpt for a node with cardinality card 
# and whose parants have cardinalities cards
def random_cpt(card,cards):
    if not cards:
        cpt = [random.uniform(0,1) for _ in range(card)]
        return np.array(cpt)/sum(cpt) # normalize
    else:
        cpt = [random_cpt(card,cards[1:]) for _ in range(cards[0])]
        return np.array(cpt)


def create_BN():
    """ create example BN A -> C <- B """
    card = 2
    bn = TBN("example1")

    a_values = ['a', '~a']
    a_cpt = random_cpt(2,[])
    A = Node("A", values=a_values, parents=[], cpt=a_cpt)
    bn.add(A) 

    b_values = ['b', '~b']
    b_cpt = random_cpt(2,[2])
    B = Node("B", values=b_values, parents=[A], cpt=b_cpt)
    bn.add(B)

    c_values = ['c', '~c']
    c_cpt = random_cpt(2, [2, 2])
    C = Node("C", values=c_values, parents=[A,B], cpt=c_cpt)
    bn.add(C)
    return bn


# Compute marginal on single output variable
# Safe VE: construct joint then sum out all other variables
def marginal(bn, output):
    nodes = bn.nodes
    qnode = bn.node(output)

    node2var = {} # map bn node to Var
    for n in nodes:
        var = VE.Var(bn_node=n)
        node2var[n] = var

    factors = [] # factor pool
    for n in nodes:
        family = n.family # cpt is over family
        vars = list(map(lambda x: node2var[x], family)) # map family to Vars
        f = VE.Factor(n.tabular_cpt(), vars, sort=True)
        factors.append(f)

    # construct joint then sum out all but qvar
    joint = VE.Factor.one()
    for f in factors:
        joint = joint.multiply(f)

    # project joint to qvar
    qvar = node2var[qnode]
    marginal = joint.project([qvar])
    marginal = marginal.table
    assert np.allclose(1.,np.sum(marginal)) # marginal is normalized
    return marginal

# Compute marginals of every node in bn
# return: marginals in the order of outputs
def node_marginals(bn):
    nodes = bn.nodes
    outputs = list(map(lambda x:x.name, nodes))
    marginals = [marginal(bn, out) for out in outputs]
    return marginals, outputs

# Compute marginals on multiple output variables
def marginal2(self, outputs):
    nodes = bn.nodes
    qnodes = [bn.node(out) for out in outputs]

    node2var = {} # map bn node to Var
    for n in nodes:
        var = VE.Var(bn_node=n)
        node2var[n] = var

    factors = [] # factor pool
    for n in nodes:
        family = n.family # cpt is over family
        vars = list(map(lambda x: node2var[x], family)) # map family to Vars
        f = VE.Factor(n.tabular_cpt(), vars, sort=True)
        factors.append(f)

    # construct joint then sum out all but qvar
    joint = VE.Factor.one()
    for f in factors:
        joint = joint.multiply(f)

    # project joint to qvar
    qvars = list(map(lambda x: node2var[x], qnodes))
    marginal = joint.project(qvars)

    # sort factor in the order of qvars
    table, mvars = marginal.table, marginal.vars
    assert set(mvars) == set(qvars)
    axes = tuple(mvars.index(v) for v in qvars)
    table = np.transpose(table, axes=axes)

    assert np.isclose(1.0, np.sum(table)) # check marginal is normalized 
    return table






    
        








if __name__ == '__main__':
    random.seed(2048)
    np.set_printoptions(precision=4)
    bn = create_BN()
    vars = ['A', 'B', 'C']
    print("CPTs: --------------------------------------------")
    for var in vars:
        node = bn.node(var)
        cpt = node.cpt
        print("{}: {}".format(var,cpt))

    marginals, outputs = node_marginals(bn)
    print("Node marginals: --------------------------------------")
    for var,mar in zip(outputs, marginals):
        print("{}: {}".format(var, mar))

    outputs = ['A', 'B']
    mar = marginal2(bn, outputs)
    print("marginals {}: {}".format(outputs, mar))
        




    



