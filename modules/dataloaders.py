import os
import pickle

import nltk
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from utils.nlp import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


class SentenceDataset(Dataset):
    """
    A PyTorch Dataset
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...
        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y,
                 word2idx,
                 lab2idx,
                 length=None,
                 name=None):
        super(SentenceDataset, self).__init__()
        self.data = X
        self.labels = y
        self.name = name
        self.max_index = 0
        self.word2idx = word2idx
        self.lab2idx = lab2idx
        self.data = self.twitter_preprocess()
        self.labels = self.label_transformer()


        if length is None:
            lengths = [len(x) for x in self.data]
            self.max_index = lengths.index(max(lengths))
            self.length = max(lengths)  # max length
        else:
            self.length = length

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['super', 'eagles', 'coach', 'sunday', 'oliseh',
                                    'meets', 'with', 'chelsea', "'", 's', 'victor',
                                    'moses', 'in', 'london', '<url>']
                self.target[index] = "neutral"

            the function will have to return return:
            ::
                example = [  533  3908  1387   649 38127  4118    40  1876    63   106  7959 11520
                            22   888     7     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0]
                label = 1
        """
        sample, label = self.data[index], self.labels[index]

        # transform the sample and the label,
        # in order to feed them to the model
        vec_sample = vectorize(sample, self.word2idx, self.length)

        return vec_sample, label, len(self.data[index])

    def twitter_preprocess(self):
        preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time',
                       'date', 'number'],
            annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis',
                      'censored'},
            all_caps_tag="wrap",
            fix_text=True,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

        text = self.data
        cache_file = os.path.join('./', "cached",
                                  "preprocessed_" + self.name + ".pkl")
        preprocessed = None
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                preprocessed = pickle.load(f)
        else:
            preprocessed = [preprocessor.pre_process_doc(x)
                            for x in tqdm(text, desc="Preprocessing dataset...")]
            with open(cache_file, 'wb') as f:
                pickle.dump(preprocessed, f)

        return preprocessed

    def label_transformer(self):
        self.y = self.labels
        y_transformed = [self.lab2idx[label] for label in self.y]
        return y_transformed
