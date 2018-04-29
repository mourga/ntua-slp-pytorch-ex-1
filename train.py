# download from http://nlp.stanford.edu/data/glove.twitter.27B.zip
# WORD_VECTORS = "../embeddings/glove.twitter.27B.50d.txt"
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch import nn
from modules.dataloaders import SentenceDataset
from modules.models import BaselineLSTMModel, AttentionalLSTM
from utils.load_embeddings import load_word_vectors
from utils.load_data import *
from utils.utilities import *
########################################################
# PARAMETERS
########################################################
from utils.utilities import label_mapping

EMBEDDINGS = "embeddings/glove.twitter.27B.50d.txt"
EMB_DIM = 50
HID_DIM = 100
BATCH_SIZE = 512
EPOCHS = 50
max_length = 40
print(torch.cuda.is_available())
########################################################
# Define the datasets/dataloaders
########################################################

# 1 - load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)
# embeddings: numpy array vocab_size x emb_size

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

# model = BaselineLSTMModel(embeddings,
#                           hidden_dim=HID_DIM,
#                           output_size=len(lab2idx),
#                           dropout_emb=0.3,
#                           dropout_lstm=0.5)

model = AttentionalLSTM(embeddings,
                        hidden_dim=HID_DIM,
                        output_size=len(lab2idx),
                        dropout_emb=0.3,
                        dropout_lstm=0.5)
# if torch.cuda.is_available():
#     model.cuda(1)

loss_function = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters)

for epoch in range(1, EPOCHS + 1):
    #############################################################################
    # Train
    #############################################################################
    train_loss = train_epoch(epoch, loader_train, model, loss_function, optimizer,
                             BATCH_SIZE, train_set)

    #############################################################################
    # Test
    #############################################################################

    val_loss, y, y_pred = test_epoch(epoch, loader_test, model, loss_function)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    print("\tTest: train loss={:.4f}, val loss={:.4f}, acc={:.4f}, f1={:.4f}".format(train_loss,
                                                                                     val_loss,
                                                                                     accuracy,
                                                                                     f1))
