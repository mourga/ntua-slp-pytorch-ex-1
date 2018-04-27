from torch import nn, torch
import torch.nn.functional as F


class BaselineLSTMModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_size):
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
        trainable_emb = False
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(num_embeddings=embeddings.shape[0],
                                      embedding_dim=embeddings.shape[1])
        self.init_embeddings(embeddings, trainable_emb)
        # the dropout "layer" for the word embeddings
        self.drop_emb = nn.Dropout(0.2)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_dim, batch_first=True, dropout=0.2)
        self.drop_rnn = nn.Dropout(0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim, output_size)

    def init_embeddings(self, weights, trainable):
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        embeds = self.word_embeddings(x)
        embeds = self.drop_emb(embeds)

        lstm_out, _ = self.lstm(embeds)
        idx = (lengths - 1).view(-1, 1).expand(lstm_out.size(0),
                                               lstm_out.size(2)).unsqueeze(1)
        last_outputs = torch.gather(lstm_out, 1, idx).squeeze()
        last_outputs = self.drop_rnn(last_outputs)
        logits = self.hidden2output(last_outputs)

        return logits
