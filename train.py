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
BATCH_SIZE = 128
EPOCHS = 50
use_gpu = torch.cuda.is_available()
print(use_gpu)
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
                 length=50,
                 name='train')

test_set = SentenceDataset(X_test, y_test,
                 word2idx,
                 lab2idx,
                 length=50,
                 name='test')

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

# define a simple model, loss function and optimizer

model = BaselineLSTMModel(embedding_dim=EMB_DIM, hidden_dim=HID_DIM,
                       vocab_size=len(word2idx), output_size=len(lab2idx)).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

#############################################################################
# Training Pipeline
#############################################################################

# loop the dataset with the dataloader that you defined and train the model
# for each batch return by the dataloader

train_loss_ = []
test_loss_ = []
train_acc_ = []
test_acc_ = []
### training procedure
for epoch in range(EPOCHS):
    ## training epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for iteration, batch in enumerate(loader_train, 1):
        # train_inputs, train_labels = traindata
        train_inputs = Variable(batch[0])
        train_labels = Variable(batch[1])
        # FIX TRAIN_LABELS
        train_labels = torch.squeeze(train_labels)

        if use_gpu:
            train_inputs = train_inputs.cuda()
            train_labels = train_labels.cuda()
        print('prothermainoume to fourno stous 180...')
        model.zero_grad()
        print("zero grad to model done")
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        print('twra tha tupwsw output')
        output = model(train_inputs)
        print('Perasame to Output')
        loss = loss_function(output, train_labels)
        print('perimenoume loss....')
        loss.backward()
        print('kaname backward......')
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels).sum()
        total += len(train_labels)
        total_loss += loss.data[0]
        print(total_loss)
    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)
    print("Epoch: ", iter, " train loss: ", total_loss / total, " accuracy ", total_acc / total)

#     ## testing epoch
#     total_acc = 0.0
#     total_loss = 0.0
#     total = 0.0
#     for iter, testdata in enumerate(test_loader):
#         test_inputs, test_labels = testdata
#         test_labels = torch.squeeze(test_labels)
#
#         if use_gpu:
#             test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
#         else:
#             test_inputs = Variable(test_inputs)
#
#         model.batch_size = len(test_labels)
#         model.hidden = model.init_hidden()
#         output = model(test_inputs.t())
#
#         loss = loss_function(output, Variable(test_labels))
#
#         # calc testing acc
#         _, predicted = torch.max(output.data, 1)
#         total_acc += (predicted == test_labels).sum()
#         total += len(test_labels)
#         total_loss += loss.data[0]
#     test_loss_.append(total_loss / total)
#     test_acc_.append(total_acc / total)
#
#     print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
#           % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))
#
# param = {}
# param['lr'] = learning_rate
# param['batch size'] = batch_size
# param['embedding dim'] = embedding_dim
# param['hidden dim'] = hidden_dim
# param['sentence len'] = sentence_len

# result = {}
# result['train loss'] = train_loss_
# result['test loss'] = test_loss_
# result['train acc'] = train_acc_
# result['test acc'] = test_acc_
# result['param'] = param
#
# if use_plot:
#     import PlotFigure as PF
#
#     PF.PlotFigure(result, use_save)
# if use_save:
#     filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
#     result['filename'] = filename
#
#     fp = open(filename, 'wb')
#     pickle.dump(result, fp)
#     fp.close()
#     print('File %s is saved.' % filename)