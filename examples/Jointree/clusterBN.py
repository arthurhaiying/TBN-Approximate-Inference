from examples.Jointree.clusterNode import ClusterNode
from utils import VE

import numpy as np


class ClusterBN:

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.name2node = {} # map name to bn node
        self.prime_var_dict = {} # map prime variable to Var

    def add(self, node):
        # add node to clusterBNu7
        if type(node) != ClusterNode:
            raise ValueError("node should be ClusterNode object")
        if not all(n in self.nodes for n in node.parents):
            raise ValueError("parents should be added before child")

        self.nodes.append(node)
        self.name2node[node.name] = node
        if node.prime is not None:
            prime = node.prime
            assert prime not in self.prime_var_dict.keys() # each clusterNode should have unique prime
            self.prime_var_dict[prime] = node.prime_var  # add prime from node

    def node(self, name):
        # return node by name
        return self.name2node[name]

    # Compute marginal on single prime variable
    def prime_marginal(self, output):
        if output not in self.prime_var_dict.keys():
            raise ValueError("Output must be a prime variable")

        # construct join and sum out all but this prime variable
        qvar = self.prime_var_dict[output]
        factors = [n.get_cluster_cpt_factor() for n in self.nodes] # factor pool
        joint = VE.Factor.one()
        for f in factors:
            joint = joint.multiply(f)
        marginal = joint.project([qvar]) 
        marginal = marginal.table
        assert np.isclose(1.0, np.sum(marginal))
        return marginal

        
        



        
        

        