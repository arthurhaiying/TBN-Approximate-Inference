from sympy import true
from tbn.node import Node

"""
    Testing nodes whose CPT are selected directly based on evidence instantiations
"""

class TestNode(Node):

    def __init__(self,name,values,parents,cpt=None,**kwargs):

        super(TestNode, self).__init__(name,values=values,parents=parents,cpt=cpt, **kwargs)                             
        self._testing_by_evd = True

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
        for attr in Node.user_attributes:
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
        node._sel_evidence = self._sel_evidence # to be replaced by corresponding nodes in net copy
        node._Node__prepare_for_inference()
        node._for_inference = True
        return node

                                

                                                                                                                                                                                                                                                                                                                               