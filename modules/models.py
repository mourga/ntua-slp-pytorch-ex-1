import torch
from torch import nn, autograd
import torch.nn.functional as F


class BaselineLSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        """
        Define the layers and initialize them.

        Pytorch initializes the layers by default, with random weights,
        sampled from certain distribution. However, in some cases
        you might want to explicitly initialize some layers,
        either by sampling from a different distribution,
        or by using pretrained weights (word embeddings / transfer learning)

        Args:
        """
        super(BaselineLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # the first is the hidden h
        # the second is the cell c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))



    def forward(self, sentence):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2output(lstm_out.view(len(sentence), -1))
        output_scores = F.log_softmax(y, dim=1)
        return output_scores