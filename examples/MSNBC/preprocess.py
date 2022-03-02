from collections import defaultdict
from pathlib import Path
import numpy as np
import csv


SEED = 2048
data_dir = Path(__file__).resolve().parent
raw_data_filename = "msnbc990928.seq"
train_data_filename = "msnbc.train.csv"
test_data_filename = "msnbc.test.csv"


def read_msnbc_dataset(filepath):
    """ Read MSNBC dataset """
    with open(filepath, 'r') as file:
        line = file.readline()
        topics = line.strip().split()
        data = []
        for line in file:
            seq = line.strip().split()
            seq = [int(x) for x in seq]
            data.append(seq)

    return data, topics

def get_X_and_Y(data, seq_len):
    """ Create input and output for predicting the next visit
        given seq_len historical visits """
    X, Y = [], []
    for seq in data:
        if len(seq) < seq_len:
            pass
        else:
            seq = seq[:seq_len]
            x_, y_ = seq[:-1], seq[-1]
            X.append(x_)
            Y.append(y_)

    return np.array(X), np.array(Y)

def train_test_split(X, Y, test_ratio=0.2, shuffle=False):
    """ Split X and Y into random train and test subsets """
    n_samples = len(X)
    n_test = int(n_samples*test_ratio)
    n_train = n_samples-n_test
    ids = list(range(n_samples))
    shuffled_ids = ids
    if shuffle: # shuffle ids
        rng = np.random.RandomState(SEED)
        shuffled_ids = rng.permutation(ids)
    train_ids = shuffled_ids[:n_train]
    test_ids = shuffled_ids[n_train:]
    X_train, X_test = X[train_ids], X[test_ids]
    Y_train, Y_test = Y[train_ids], Y[test_ids]
    return X_train, X_test, Y_train, Y_test


def create_dataset(seq_len, filename, train_filename, test_filename):
    filepath = data_dir / filename
    data, topics = read_msnbc_dataset(filepath)
    X, Y = get_X_and_Y(data, seq_len)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    # save training set
    train_filepath = data_dir / train_filename
    save_dataset(train_filepath, X_train, Y_train)
    # save testing set
    test_filepath = data_dir / test_filename
    save_dataset(test_filepath, X_test, Y_test)
    # save topics
    topic_filepath = data_dir / "topic.txt"
    save_topics(topic_filepath, topics)
    return X_train, X_test, Y_train, Y_test, topics

def save_topics(filepath, topics):
    with open(filepath, 'w') as file:
        file.write(','.join(topics))
        file.write('\n')

def load_topics(filepath):
    with open(filepath, 'r') as file:
        line = file.readline()
        topics = line.strip().split(',')
        return topics

def save_dataset(filepath, X, Y):
    seq_len = X.shape[1]
    fields = ['X%d'%i for i in range(seq_len)] + ['Y']
    arr = np.concatenate((X, Y.reshape(-1,1)), axis=-1)
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(arr.tolist())


def load_dataset(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        arr = list(reader)[1:] # skip first line
        arr = np.array(arr, dtype=int)
        X, Y = arr[:,:-1], arr[:, -1]
    return X, Y






if __name__ == '__main__':
    seq_len = 12
    X_train, X_test, Y_train, Y_test, topics = create_dataset(seq_len,
                                                      raw_data_filename,
                                                      train_data_filename,
                                                      test_data_filename)
    n_train, n_test = len(X_train), len(X_test)
    print("Finish creating datasets.")
    print("n_train: {} n_test: {}".format(n_train, n_test))























