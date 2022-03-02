from approximate.networks.NetParser import parseBN
from approximate.jointree import make_natural_join_tree

def main():
    net_filepath = "approximate/networks/asia2.net"
    bn = parseBN(net_filepath)
    T, clusters = make_natural_join_tree(bn, trim=True)
