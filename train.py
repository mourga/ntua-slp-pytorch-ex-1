# download from http://nlp.stanford.edu/data/glove.twitter.27B.zip
# WORD_VECTORS = "../embeddings/glove.twitter.27B.50d.txt"
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn, optim
from modules.dataloaders import SentenceDataset
from modules.models import BaselineLSTMModel
from utils.load_embeddings import load_word_vectors
from utils.load_data import *
import numpy as np

########################################################
# PARAMETERS
########################################################
from utils.utilities import label_mapping

EMBEDDINGS = "embeddings/glove.twitter.27B.50d.txt"
EMB_DIM = 50
HID_DIM = 100
BATCH_SIZE = 16
EPOCHS = 50
max_length = 40
print(torch.cuda.is_available())
########################################################
# Define the datasets/dataloaders
########################################################

# 1 - load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# you can load the raw data like this:
train = load_semeval2017A("datasets/Semeval2017A/train_dev")
val = load_semeval2017A("datasets/Semeval2017A/gold")

X_train = [key[1] for key in train]
y_train = [key[0] for key in train]

X_test = [key[1] for key in val]
y_test = [key[0] for key in val]

lab2idx, idx2lab = label_mapping(y_train)

# 2 - define the datasets
train_set = SentenceDataset(X_train, y_train,
                            word2idx,
                            lab2idx,
                            length=max_length,
                            name='train')

test_set = SentenceDataset(X_test, y_test,
                           word2idx,
                           lab2idx,
                           length=max_length,
                           name='test')

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

# define a simple model, loss function and optimizer

model = BaselineLSTMModel(embeddings, HID_DIM, len(lab2idx))
if torch.cuda.is_available():
    model.cuda(1)

loss_function = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters)


def train_epoch(_epoch, data_loader, _model, _loss_function):
    model.train()
    ## training epoch
    total_loss = 0.0
    print('epoch', epoch)
    for iteration, batch in enumerate(data_loader, 1):
        samples, labels, lengths = batch
        samples = Variable(samples)
        labels = Variable(labels)
        lengths = Variable(lengths)

        if torch.cuda.is_available():
            samples = samples.cuda(1)
            labels = labels.cuda(1)
            lengths = lengths.cuda(1)

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass
        output = model(samples, lengths)
        # output_new = output.cpu()
        # 3 - compute loss
        loss = loss_function(output, labels)

        # 4 - backward pass
        loss.backward()

        # 5 - optimizer step
        optimizer.step()

        total_loss += loss.data[0]

    return total_loss

#############################################################################
# Train
#############################################################################

# loop the dataset with the dataloader that you defined and train the model
# for each batch return by the dataloader

# acc = lambda y, y_hat: accuracy_score(y, y_hat)
# f1 = lambda y, y_hat: f1_score(y, y_hat, average='macro')

for epoch in range(1, EPOCHS + 1):
    print(train_epoch(epoch, loader_train, model, loss_function))


