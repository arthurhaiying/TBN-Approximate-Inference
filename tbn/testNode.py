from copy import copy
import numpy as np

from tbn.node import Node
import tbn.cpt
import utils.utils as u

"""
    Testing nodes whose CPT are selected directly based on evidence instantiations
"""

class TestNode(Node):

    # user attributes are ones that can be specified by the user when constructing node
    user_attributes = ('name','values','parents','num_cpts','cpt','cpts','cpt_tie')

    def __init__(self,name,*,values,parents,num_cpts=1,cpt=None,cpts=None,cpt_tie=None):
        
        super(TestNode, self).__init__(name,values=values,parents=parents,cpt=cpt,cpt_tie=cpt_tie)                             
        self._testing_by_evd = True
        self._sel_evidence   = None
        # check validity of cpt arguments
        assert num_cpts >= 1
        if num_cpts == 1:
            if cpts is not None: 
                raise ValueError("Node cannot have cpts if num_cpts is one.")
        else: #num_cpts >= 2:
            if cpts is not None and len(cpts) != num_cpts:
                raise ValueError("cpts do not match num_cpts.")
        cpts  = copy(cpts)
        card  = len(values)
        cards = tuple(p.card for p in parents)
        if num_cpts >= 2 and cpts is None:
            if cpt is not None:
                cpts = [cpt]*num_cpts
            else:
                cpts = [tbn.cpt.random(card,cards) for _ in range(num_cpts)]

        self._num_cpts = num_cpts
        self._cpts     = cpts


    @property
    def num_cpts(self):  return self._num_cpts
    @property
    def cpts(self):  return self._cpts
    @property
    def sel_evidence(self):  return self._sel_evidence
    @property
    def cpt1(self): # cpt1 now points to first cpt in cpts
        if self.num_cpts == 2:
            return self.cpts[0]
        else:
            return self._cpt1
    @property
    def cpt2(self): # cpt2 now points to second cpt in cpts
        if self.num_cpts == 2:
            return self.cpts[1]
        else:
            return self._cpt2
    

    # assign selection evidence after all TBN nodes are added
    def set_selection_evidence(self, evidence):
        assert self._testing_by_evd
        assert self.tbn is not None
        assert len(evidence) >= 1
        for e in evidence:
            if not isinstance(e, Node):
                raise ValueError(f"Selection evidence {e} should be TBN Node")
            if e.name not in self.tbn._n2o.keys():
                raise ValueError(f"Evidence {e} of node {self.name} has not been  \
                                   added to TBN {self.tbn.name}")
        self._sel_evidence = evidence


    # override copy_for_inference method
    def copy_for_inference(self,tbn):
        assert not self._for_inference
        kwargs = {}
        dict   = self.__dict__
        for attr in TestNode.user_attributes: # including num_cpts and cpts
            _attr = f'_{attr}'
            assert _attr in dict
            value = dict[_attr]
            if attr=='parents': 
                value = [tbn.node(n.name) for n in value]
            kwargs[attr] = value 
        # node has the same user attribues as self except that 
        #   1) parents of self are replaced by corresponding nodes in tbn
        #   2) sel evidence of self need to be replaced by corresponding nodes in tbn
        node = TestNode(**kwargs) # testing_by_evd
        node._sel_evidence = self._sel_evidence # to be replaced by corresponding nodes in copied net
        node.__prepare_for_inference()
        node._for_inference = True
        return node

    # -prunes node values and single-value parents
    # -expands cpts into np arrays
    # -identifies 0/1 cpts
    # -sets cpt labels (for saving into file)
    # -sorts parents, family and cpts
    def __prepare_for_inference(self):
    
        # the following attributes are updated in decouple.py, which replicates
        # functional cpts and handles nodes with hard evidence, creating clones
        # of nodes in the process (clones are added to another 'decoupled' network)
        self._original   = None  # tbn node cloned by this one
        self._master     = None  # exactly one clone is declared as master
        self._clamped    = False # whether tbn node has hard evidence
        
        # the following attributes with _cpt, _cpt1, _cpt2 are updated in cpt.y
        self._values_org = self.values # original node values before pruning
        self._card_org   = self.card   # original node cardinality before pruning
        self._values_idx = None        # indices of unpruned values, if pruning happens
        
        # -process node and its cpts
        # -prune node values & parents and expand/prune cpts into tabular form  
        tbn.cpt.set_cpts2(self)
                
        # the following attributes will be updated next
        self._all01_cpt  = None  # whether cpt is 0/1 (not applicable for testing nodes)
        self._cpt_label  = None  # for saving to file (updated when processing cpts)
        
        # tested cpt is not necessarily all zero-one even if cpt1 and cpt2 are
        self._all01_cpt = False
       
        # -pruning node values or parents changes the shape of cpt for node
        # -a set of tied cpts may end up having different shapes due to pruning
        # -we create refined ties between groups that continue to have the same shape
        """ this is not really proper and needs to be updated """
        if self.cpt_tie is not None:
#            s = '.'.join([str(hash(n.values)) for n in self.family])
            self._cpt_tie = f'{self.cpt_tie}__{self.shape()}'
            
        self.__set_cpt_labels()
        
        # we need to sort parents & family and also adjust the cpt accordingly
        # this must be done after processing cpts which may prune parents
        self.__sort()
        assert u.sorted(u.map('id',self.parents))
        assert u.sorted(u.map('id',self.family))
        
    
    # sort family and reshape cpt accordingly (important for ops_graph)
    def __sort(self):
        assert type(self.parents) is list and type(self.family) is list
        
        if u.sorted(u.map('id',self.family)): # already sorted
            self._parents = tuple(self.parents)
            self._family  = tuple(self.family)
            return
        
        self._parents.sort()
        self._parents = tuple(self.parents)
        
        # save original order of nodes in family (needed for transposing cpt)
        original_order = [(n.id,i) for i,n in enumerate(self.family)]
        self.family.sort()
        self._family = tuple(self.family)
        
        # sort cpt to match sorted family
        original_order.sort() # by node id to match sorted family
        sorted_axes = [i for (_,i) in original_order] # new order of axes
        if self.num_cpts == 1:
            self._cpt = np.transpose(self.cpt,sorted_axes)
        else: # self.num_cpts >= 2
            for i in range(self.num_cpts):
                cpt = self.cpts[i]
                self._cpts[i] = np.transpose(cpt,sorted_axes)
                

    # sets cpt labels used for saving cpts to file
    def __set_cpt_labels(self):
        
        # maps cpt type to label
        self._cpt_label = {}
        def set_label(cpt,cpt_type):
            assert cpt_type not in self.cpt_label
            type_str    = cpt_type + (f' (tie_id {self.cpt_tie})' if self.cpt_tie else '')
            parents_str = u.unpack(self.parents,'name')
            self._cpt_label[cpt_type] = f'test {type_str}: {self.name} | {parents_str}'
        
        set_label(self.cpt, 'cpt')
        if self.num_cpts >= 2:
            for i in range(self.num_cpts):
                type = f'cpt{i+1}'
                cpt = self.cpts[i]
                set_label(cpt, type)
                




                                

                                                                                                                                                                                                                                                                                                                               