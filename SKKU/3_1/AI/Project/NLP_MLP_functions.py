# -*- coding: utf-8 -*-

import json # import json module
import numpy as np
import csv
import pickle
import math
import codecs
import copy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0) #protection from oveflow
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def val_train_set(train_data,train_labels):
    total_len=len(train_data)
    idx = np.arange(0, total_len)
    np.random.shuffle(idx)
    val_idx = idx[:int(total_len * 0.2)]
    validation_data = [train_data[i] for i in val_idx]
    validation_label = [train_labels[i] for i in val_idx]
    train_idx = idx[int(total_len * 0.2):]
    train_data_set = [train_data[i] for i in train_idx]
    train_label_set = [train_labels[i] for i in train_idx]
    return validation_data, validation_label, train_data_set, train_label_set
