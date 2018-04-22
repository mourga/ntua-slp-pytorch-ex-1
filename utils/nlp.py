import numpy as np


def tokenize(text, lowercase=True):
    pass


def vectorize(text, word2idx, max_length):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length (): the maximum length of the input sequences

    Returns: zero-padded list of ids

    """

    text = text[:max_length]
    vec = np.zeros(max_length, dtype=int)
    for i, word in enumerate(text):
        if word in word2idx.keys():
            vec[i] = word2idx[word]
        else:
            vec[i] = word2idx['<unk>']
    return vec
