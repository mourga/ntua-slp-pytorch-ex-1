import glob
import os
import pickle
import numpy as np

def file_cache_name(file):
    head, tail = os.path.split(file)
    filename, ext = os.path.splitext(tail)
    return os.path.join(head, filename + ".p")


def write_cache_word_vectors(file, data):
    with open(file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_cache_word_vectors(file):
    with open(file_cache_name(file), 'rb') as f:
        return pickle.load(f)

def label_mapping(y):
    y_np = np.array(y)
    un_labels = np.unique(y_np)
    lab2idx = {}
    idx2lab = {}
    for idx, lab in enumerate(un_labels):
        lab2idx[lab] = idx
        idx2lab[idx] = lab
    return lab2idx, idx2lab

