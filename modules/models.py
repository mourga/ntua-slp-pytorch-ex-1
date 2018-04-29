from torch import nn, torch
import torch.nn.functional as F

from modules.layers import Attention


class BaselineLSTMModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_size, dropout_emb, dropout_lstm):
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

        # 1) embedding layer:
        trainable_emb = False
        self.word_embeddings = nn.Embedding(num_embeddings=embeddings.shape[0],
                                            embedding_dim=embeddings.shape[1])
        self.init_embeddings(embeddings, trainable_emb)
        self.drop_emb = nn.Dropout(dropout_emb)

        # 2) LSTM layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_dim, batch_first=True,
                            dropout=dropout_lstm)
        self.drop_lstm = nn.Dropout(dropout_lstm)

        # 3) linear layer -> outputs
        self.hidden2output = nn.Linear(hidden_dim, output_size)

    def init_embeddings(self, weights, trainable):
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(weights),
                                                   requires_grad=trainable)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        x: batch_size x max_length (batch_size tweets from Dataloader)

        """
        embeds = self.word_embeddings(x)
        embeds = self.drop_emb(embeds)

        lstm_out, _ = self.lstm(embeds)
        idx = (lengths - 1).view(-1, 1).expand(lstm_out.size(0),
                                               lstm_out.size(2)).unsqueeze(1)
        last_outputs = torch.gather(lstm_out, 1, idx).squeeze()
        last_outputs = self.drop_lstm(last_outputs)
        logits = self.hidden2output(last_outputs)

        return logits


class AttentionalLSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_size, dropout_emb, dropout_lstm):
        """
        Define the layers and initialize them.

        Pytorch initializes the layers by default, with random weights,
        sampled from certain distribution. However, in some cases
        you might want to explicitly initialize some layers,
        either by sampling from a different distribution,
        or by using pretrained weights (word embeddings / transfer learning)

        Args:
        """
        super(AttentionalLSTM, self).__init__()

        # 1) embedding layer:
        trainable_emb = False
        self.word_embeddings = nn.Embedding(num_embeddings=embeddings.shape[0],
                                            embedding_dim=embeddings.shape[1])
        self.init_embeddings(embeddings, trainable_emb)
        self.drop_emb = nn.Dropout(dropout_emb)

        # 2) LSTM layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embeddings.shape[1],
                            hidden_size=hidden_dim,
                            batch_first=True,
                            dropout=dropout_lstm)
        self.drop_lstm = nn.Dropout(dropout_lstm)
        self.attention = Attention(attention_size=hidden_dim, batch_first=True)

        # 3) linear layer -> outputs
        self.hidden2output = nn.Linear(hidden_dim, output_size)

    def init_embeddings(self, weights, trainable):
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(weights),
                                                   requires_grad=trainable)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        x: batch_size x max_length (batch_size tweets from Dataloader)

        """
        embeds = self.word_embeddings(x)
        embeds = self.drop_emb(embeds)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.drop_lstm(lstm_out)

        # idx = (lengths - 1).view(-1, 1).expand(lstm_out.size(0),
        #                                        lstm_out.size(2)).unsqueeze(1)
        # last_outputs = torch.gather(lstm_out, 1, idx).squeeze()
        # last_outputs = self.drop_lstm(last_outputs)
        representations, scores = self.attention(lstm_out, lengths)
        logits = self.hidden2output(representations)

        return logits
