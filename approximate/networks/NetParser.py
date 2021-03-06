import ast
import numpy as np
from pathlib import Path
import re

from tbn.tbn import TBN
from tbn.node import Node

def getline(f):
    for line in f:
        line = line.strip()
        if line:
            # return the first non blank line
            return line
    return "EOF"

def str2arr(str):
    str=str.strip()
    str=re.sub(r'\(','[',str)
    str=re.sub(r'\)',']',str)
    str=re.sub(r'(\d+)\s+(\d+)',r'\1,\2',str)
    str=re.sub(r'\]\s*\[','],[',str)
    #print(str)
    arr = np.array(ast.literal_eval(str))
    return arr

# return a topological ordering of nodes in dag
# dag: a dict of list representing directed adjacencies of DAG
def topo_sort(dag):

    in_degrees = {node:0 for node in dag.keys()}

    # compute in-degrees of each node in dag
    for node in dag.keys():
        for child in dag[node]:
            in_degrees[child]+=1
    
    queue = [node for node in dag.keys() if in_degrees[node] == 0] # roots
    order = []
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for child in dag[curr]:
            in_degrees[child]-=1
            # decrement in-degrees for each child
            if in_degrees[child] == 0:
                queue.append(child)

    return order

def parseBN(fname):
    f = open(fname,mode='r')
    bn_name = Path(fname).name.split('.')[0]
    node_pattern = re.compile(r"node\s+(\S+)")
    state_pattern = re.compile(r'"(.*?)"')
    states_pattern = re.compile(r"states\s*=\s*\((.*)\)\s*;$")
    potential_pattern = re.compile(r"potential\s*\((.*)\)")
    data_pattern = re.compile(r"data\s*=(.*);$")

    nodes = []
    states_dict,parents_dict,potential_dict = {},{},{}
    dag = {}

    line = getline(f)
    while line != 'EOF':
        if line.startswith("node"):
            # match node
            m = node_pattern.match(line)
            assert m
            name = m.group(1)
            nodes.append(name)
            assert getline(f) == '{' # skip 
            line = getline(f)
            m = states_pattern.match(line)
            assert m # match states
            states = re.findall(state_pattern,m.group())
            assert len(states) >= 2 # node has at least two states
            states_dict[name] = states
            assert getline(f) == '}' 
        
        elif line.startswith("potential"):
            m = potential_pattern.match(line)
            # match potential
            assert m
            family = m.group(1)
            index = family.find('|')
            if index == -1:
                name = family.strip()
                parents = []
            else:
                name = family[:index].strip()
                parents = family[index+1:].strip().split()
            #print("node: {} parents: {}".format(name,parents))
            parents_dict[name] = parents
            assert getline(f) == '{'
            line = getline(f)
            m = data_pattern.match(line)
            arr = str2arr(m.group(1))
            potential_dict[name] = arr
            assert getline(f) == '}'

        #else:
            #raise RuntimeError("Unrecognized line: %s" %(line,))

        line = getline(f) 
        # read next line


    # check nodes, states, parents, and potentials
    # print('nodes: %s' %nodes)
    # print('parent dict: %s' %parents_dict)
    dag = {node:[] for node in nodes}
    for node in nodes:
        shape = []
        parents = parents_dict[node]
        #print("node: %s parents: %s" % (node, parents))
        for p in parents:
            assert p in nodes # parent exists
            dag[p].append(node) # add edge from p to node
            shape.append(len(states_dict[p]))
        
        shape.append(len(states_dict[node]))
        potential = potential_dict[node]
        assert potential.shape == tuple(shape)
        assert np.allclose(np.sum(potential,axis=-1),1.0) 
        # check valid potential

    # make bn object
    bn = TBN(bn_name)
    order = topo_sort(dag)
    for node in order:
        name = node
        values = states_dict[node]
        parents = list(map(bn.node, parents_dict[node]))
        potential = potential_dict[node]
        bnNode = Node(name=name,values=values,parents=parents,testing=False,cpt=potential)
        bn.add(bnNode)

    f.close()
    return bn


def writeBN(bn, fname):
    f = open(fname,'w+')

    def __write_node(node):
        name,states = node.name,node.values
        states_pattern = ['"%s"' % state for state in states]
        states_pattern = " ".join(states_pattern)
        f.write("node %s\n" % name)
        f.write("{\n")
        f.write("   states = ( %s );\n" % states_pattern)
        f.write("}\n")

    def __write_potential(node):
        name,parents,cpt = node.name,node.parents,node.cpt
        parents = [p.name for p in parents]
        parents_pattern = " ".join(parents)
        cpt_pattern = np.array2string(cpt,precision=6)
        cpt_pattern = cpt_pattern.replace('\n', '')
        cpt_pattern = re.sub(r'\[','(',cpt_pattern)
        cpt_pattern = re.sub(r'\]',')',cpt_pattern)
        f.write("potential ( %s | %s )\n" %(name, parents_pattern))
        f.write("{\n")
        f.write("   data = %s;\n" % cpt_pattern)
        f.write("}\n")
    
    for node in bn.nodes:
        __write_node(node)
        # write nodes
    for node in bn.nodes:
        __write_potential(node)
        # write potentials
    f.flush()
    f.close()
    

if __name__ == '__main__':
    filename = 'approximate/networks/asia.net'
    print("start reading net...")
    bn = parseBN(filename)
    print("finish reading net.")
    for node in bn.nodes:
        name = node.name
        values = node.values
        parents = list(map(lambda x:x.name,node.parents))
        cpt = node.cpt
        print("node: {} values: {} parents: {} ".format(name,values,parents))
        print("potential: {}".format(cpt))

    filename2 = 'approximate/networks/asia2.net'
    print("start writing net")
    writeBN(bn,filename2)
    print("Finish writing net.")








        





