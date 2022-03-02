from pathlib import Path
import sys


libpath = Path(__file__).resolve().parents[2]
#print("libpath: {}".format(libpath))
sys.path.append(str(libpath))

from tbn.tbn import TBN
from tbn.node import Node
from tac import TAC
import train.data as Data
import examples.MSNBC.preprocess as Preprocess


data_dir = Path(__file__).resolve().parent
raw_data_filename = "msnbc990928.seq"
train_data_filename = "msnbc.train.csv"
test_data_filename = "msnbc.test.csv"

def transform_data(X, Y, n_topics):
    var_count = len(X[0])
    # Be careful: index is one-based in MSNBC datasets
    X -= 1
    Y -= 1
    cards = [n_topics for _ in range(var_count)]
    X_ = Data.evd_id2col(X, cards)
    Y_ = Data.mar_id2mar(Y, n_topics)
    return X_, Y_


def create_HMM(seq_len, hidden_size, evd_size, *, testing=False, tied_params=False):
    name = "test-hmm" if testing else "hmm"
    hmm = TBN(name)
    hvalues = ['v%d'%i for i in range(hidden_size)]
    evalues = ['v%d'%i for i in range(evd_size)]
    hidden_node_cache = {} # map h_i to hidden node
    h_0 = Node("H0",values=hvalues,parents=[])
    hidden_node_cache[0] = h_0
    hmm.add(h_0)


    # add hidden nodes
    for i in range(1, seq_len):
        name = 'H%d' % i
        parent = hidden_node_cache[i-1] # h_(i-1)
        cpt_tie = "transition" if tied_params else None
        h_i = Node(name,values=hvalues,parents=[parent],testing=testing,cpt_tie=cpt_tie)
        hidden_node_cache[i] = h_i
        hmm.add(h_i)

    # add evidence nodes
    for i in range(seq_len):
        name = 'E%d' % i
        parent = hidden_node_cache[i]
        cpt_tie = "emission" if tied_params else None
        e_i = Node(name,values=evalues,parents=[parent],testing=testing,cpt_tie=cpt_tie)
        hmm.add(e_i)

    return hmm


def main(seq_len, hidden_size, tied_params, train_filename, test_filename):
    # load train and test datasets
    topic_filepath = data_dir / "topic.txt"
    topics = Preprocess.load_topics(topic_filepath)
    train_filepath = data_dir / train_filename
    X_train, Y_train = Preprocess.load_dataset(train_filepath)
    test_filepath = data_dir / test_filename
    X_test, Y_test = Preprocess.load_dataset(test_filepath)
    n_topic = len(topics)
    n_train, n_test = len(X_train), len(X_test)
    print("Load datasets: n_topic {}, n_train {}, n_test {}".format(n_topic, n_train, n_test))
    # transform data to TAC input/output format
    X_train, Y_train = transform_data(X_train, Y_train, n_topic)
    X_test, Y_test = transform_data(X_test, Y_test, n_topic)
    # create HMM/ test_hMMs
    hmm = create_HMM(seq_len,hidden_size,n_topic,testing=False,tied_params=tied_params)
    test_hmm = create_HMM(seq_len,hidden_size,n_topic,testing=True,tied_params=tied_params)
    print("Build models: seq_len {}, hidden_size {}, evd_size {} ".format(seq_len, hidden_size, n_topic))
    # compile hmm/test_hmm to AC/TAC
    inputs = ['E%d'%i for i in range(seq_len-1)]
    output = 'E%d'% (seq_len-1)
    ac = TAC(hmm,inputs, output, trainable=True)
    tac = TAC(test_hmm, inputs, output, trainable=True)
    # learn AC/TAC
    ac.fit(X_train, Y_train, loss_type='CE', metric_type='CE')
    tac.fit(X_train, Y_train, loss_type='CE', metric_type='CE')
    # test AC/TAC
    CE_loss = ac.metric(X_test, Y_test, metric_type='CE')
    test_CE_loss = tac.metric(X_test, Y_test,metric_type='CE')
    acc = ac.metric(X_test, Y_test, metric_type='CA')
    test_acc = tac.metric(X_test, Y_test, metric_type='CA')
    print("Testing CE loss: AC {:5f}, TAC {:5f}".format(CE_loss, test_CE_loss))
    print("Testing acc: AC {:5f}, TAC {:5F}".format(acc, test_acc))


if __name__ == '__main__':
    seq_len = 12
    hidden_size = 17
    tied_params = True
    main(seq_len, hidden_size,tied_params,train_data_filename,test_data_filename)





    


