
from sklearn.cluster import cluster_optics_xi
from tbn.node import Node
from tbn.tbn import TBN
from utils import VE

import numpy as np

class ClusterNode(Node):

    def __init__(self, name, values, parents, *, cpt=None, is_cluster=False, 
                 prime=None, cluster=None, cluster_cards=None):

        # check validity of arguments
        if is_cluster:
            if cluster is None or cluster_cards is None or prime is None:
                raise ValueError("Cluster node must have cluster, cluster_cards, and prime")
            if len(cluster) != len(cluster_cards):
                raise ValueError("cluster and cluster_cards must have the same length")
            flat_card = 1 
            for card in cluster_cards:
                flat_card *= card
            if len(values) != flat_card:
                raise ValueError("Number of states does not match!")

        super(ClusterNode, self).__init__(name=name, values=values, parents=parents, cpt=cpt)
        self.is_cluster = is_cluster
        self.cluster = list(cluster) if is_cluster else [name]
        self.cluster_cards = tuple(cluster_cards) if is_cluster else (len(values),)

        # set prime and sub variables  
        if not is_cluster: 
            # for regular node
            self.prime = name
            self.sub = []
        else:  
            # for cluster node
            self.prime = prime
            self.sub = [x for x in self.cluster if x != prime]


        self.recycle_bin = [] # store unused object

        # build vars for VE
        self.var = VE.Var(bn_node=self)
        self.cluster_vars = self.make_cluster_vars() 
        if not is_cluster:
            self.prime_var = self.var
        else:
            idx = self.cluster[self.prime]
            self.prime_var = self.cluster[idx]
        
    def make_cluster_vars(self):
        if not self.is_cluster: 
            return [self.var]

        vars = []
        for x,card in zip(self.cluster, self.cluster_cards):
            if x == self.prime:
                name = x
            else:
                name = "%d_in_%d" %(x, self.name)
            values = ['state%d'%i for i in range(card)]
            node = Node(name,values=values) # auxiliary bn node for var
            var = VE.Var(bn_node=node)
            vars.append(var)
            self.recycle_bin.append(node) # not sure what happen if aux node is deleted

        return vars

    
    def get_cpt_factor(self):
        vars = [node.var for node in self.family]
        factor = VE.Factor(self.tabular_cpt(), vars=vars, sort=True)
        return factor

    def get_cluster_cpt_factor(self): # get clustered cpt factor
        cpt = self.tabular_cpt()
        new_shape = ()
        for node in self.family:
            new_shape += self.cluster_cards
        cluster_cpt = cpt.reshape(new_shape)
        vars = []
        for node in self.family:
            vars.extend(node.cluster_vars) 

        factor = VE.Factor(cluster_cpt,vars=vars, sort=True)
        return factor









        
            






