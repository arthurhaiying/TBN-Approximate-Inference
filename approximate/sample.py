from multiprocessing import Pool
from pathlib import Path
import csv
import numpy as np
import time

from utils import VE
from train import data as Data
from approximate.networks import NetParser

"""
    Sample data from Bayesian Networks
"""

SAMPLE_DIR = 'approximate/data'
SAMPLE_SEED = 2048
NUM_SAMPLE_WORKERS = 10

# direct sample full variable instantiations from Bayesian network 
# data: a list of list that contains sampled instantiations
# order: the used variable order
def direct_sample(bn, sample_size, seed):
    data = []
    order = [n.name for n in bn.nodes]
    rng = np.random.RandomState(seed)
    for i in range(sample_size):
        inst = {} # map var name to value
        for node in bn.nodes: 
            # topologically sorted
            x = node.name
            parents = node.parents
            cpt = node.tabular_cpt()
            if not parents: # roots
                cond = cpt
            else: 
                parents = [p.name for p in parents]
                pvalues = [inst[p] for p in parents]
                cond = cpt[tuple(pvalues)]
            cond = cond / np.sum(cond) # normalize
            value = rng.choice(len(node.values), p=cond)
            inst[x] = value

        row = [inst[x] for x in order]
        data.append(row)

    return data, order


def direct_sample_and_save(bn, sample_size, filename=None, resample=False):
    if filename is None:
        filename = f"{bn.name}_{sample_size}.csv"
    filepath = Path(SAMPLE_DIR) / filename
    # load dataset if it already exists
    ok = False
    if not resample:
        if filepath.is_file():
            with open(filepath, 'r', newline='') as file:
                reader = csv.reader(file)
                order = next(reader)
                data = list(reader)
                data = np.array(data).astype(int)
            if len(data) == sample_size:
                ok = True
                print("Dataset already sampled.")

    # sample new dataset
    if not ok:
        start = time.time()
        data, order = direct_sample(bn, sample_size, SAMPLE_SEED)
        end = time.time()
        print(f"Dataset sampled: {end-start}s.")
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(order)
        writer.writerows(data)

    return data, order


def direct_sample_mp(bn, sample_size, seed, num_workers):
    order = [n.name for n in bn.nodes]
    chunk_size = sample_size // num_workers
    last_chunk_size = chunk_size + sample_size % num_workers
    chunk_sizes = [chunk_size] * num_workers
    chunk_sizes[-1] = last_chunk_size
    seeds = [seed+i*100 for i in range(num_workers)] 
    
    total_data = []
    args = []
    for size,seed in zip(chunk_sizes, seeds):
        args.append((bn, size, seed))
    with Pool(num_workers) as p:
        result = p.starmap(direct_sample, args)
        for data,_ in result:
            total_data.extend(data)
    return total_data, order
        
# direct sample full variable instantiations from bn
# filename: if None, use bn name and sample size
# resample: if false, will not sample again if dataset is already sampled
def direct_sample_and_save_mp(bn, sample_size, filename=None, resample=False):
    if filename is None:
        filename = f"{bn.name}_{sample_size}.csv"
    filepath = Path(SAMPLE_DIR) / filename

    # load dataset if it already exists
    ok = False
    if not resample:
        if filepath.is_file():
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                order = next(reader)
                data = list(reader)
                data = np.array(data).astype(int)
            if len(data) == sample_size:
                ok = True
                print("Dataset already sampled.")

    # sample new dataset
    if not ok:
        start = time.time()
        data, order = direct_sample_mp(bn, sample_size, SAMPLE_SEED, NUM_SAMPLE_WORKERS)
        end = time.time()
        print(f"Dataset sampled: {end-start}s.")
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(order)
        writer.writerows(data)

    return data, order




        


        

    



    


def split_data(data, order, evidences, Q):
    data = np.array(data)
    eids = [order.index(e) for e in evidences]
    evidences = data[:, eids]
    qid = order.index(Q)
    marginals = data[:, qid]
    return evidences, marginals




# estimate conditional probability Pr(Q|e) from sampled data
# evidence: a dict(var -> value) that represents instantiation of evidence variables
# Q, card: query variable and its cardinality
# data, order: sampled data and the used variable order
# Return: posterior of Q given evidence
def estimate(evidence, Q, card, data, order):
    marginal = np.zeros(card)
    data2 = [] # data rows that satisfy evidence
    for row in data:
        ok = True
        for var,value in evidence.items():
            id = order.index(var)
            if row[id] != value:
                ok = False
        if ok:
            data2.append(row)

    for row in data2:
        qid = order.index(Q)
        marginal[row[qid]] += 1

    # print("frequency: {}".format(marginal))
    marginal = marginal.astype(np.float)
    marginal = marginal / np.sum(marginal)
    return marginal
        



def main():
    net_filepath = "approximate/networks/asia.net"
    data_filename = "asia.csv"
    np.set_printoptions(precision=3)
    bn = NetParser.parseBN(net_filepath)
    evidence = {"asia":1, "smoke":1, "xray":0}
    Q = "dysp"
    card = 2
    #########  Inference by sampling ####################################
    sample_size = 1000
    data, order = direct_sample_and_save(bn, sample_size, data_filename)
    marginal = estimate(evidence, Q, card, data, order)
    ########## Inference by PyTAC #######################################
    inputs, values = list(evidence.keys()), list(evidence.values())
    evidence_ve = Data.evd_id2col([values], cards=[2,2,2])
    marginal_ve = VE.posteriors(bn, inputs=inputs, output=Q, evidence=evidence_ve)
    print("true marginal: {} sample marginal: {}".format(marginal_ve, marginal))


if __name__ == '__main__':
    main()

















