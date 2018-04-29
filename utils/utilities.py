import os
import pickle
import numpy as np
import sys
import math
from torch.autograd import Variable
import torch


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


def progress(loss, epoch, batch, batch_size, dataset_size):
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_epoch(epoch, data_loader, model, loss_function, optimizer,
                batch_size, train_set):

    model.train()

    ## training epoch
    total_loss = 0.0
    print('epoch', epoch)
    for iteration, batch in enumerate(data_loader, 1):
        samples, labels, lengths = batch
        samples = Variable(samples)
        labels = Variable(labels)
        lengths = Variable(lengths)

        # if torch.cuda.is_available():
        #     samples = samples.cuda(1)
        #     labels = labels.cuda(1)
        #     lengths = lengths.cuda(1)

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass
        output = model(samples, lengths)

        # 3 - compute loss
        _loss = loss_function(output, labels)

        # 4 - backward pass
        _loss.backward()

        # 5 - optimizer step
        optimizer.step()

        total_loss += _loss.data[0]

        # print statistics
        progress(loss=_loss.data[0],
                 epoch=epoch,
                 batch=iteration,
                 batch_size=batch_size,
                 dataset_size=len(train_set))

    _avg_loss = total_loss / iteration

    return _avg_loss


def test_epoch(epoch, data_loader, model, loss_function):

    model.eval()
    y_pred = []
    y = []

    total_loss = 0
    print('epoch', epoch)
    for iteration, batch in enumerate(data_loader, 1):
        samples, labels, lengths = batch
        samples = Variable(samples, volatile=True)
        labels = Variable(labels, volatile=True)
        lengths = Variable(lengths, volatile=True)

        if torch.cuda.is_available():
            samples = samples.cuda(1)
            labels = labels.cuda(1)
            lengths = lengths.cuda(1)

        # 1 - zero the gradients
        # optimizer.zero_grad()

        # 2 - forward pass
        output = model(samples, lengths)

        # 3 - compute loss
        _loss = loss_function(output, labels)

        # 4 - backward pass
        # loss.backward()

        # 5 - optimizer step
        # optimizer.step()

        total_loss += _loss.data[0]

        _, predicted = torch.max(output.data, 1)
        y.extend(list(labels.data.cpu().numpy().squeeze()))
        y_pred.extend(list(predicted.squeeze()))

    _avg_loss = total_loss / iteration

    return _avg_loss, y, y_pred
