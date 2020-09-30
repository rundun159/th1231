# 은닉층의 계층 수를 늘리면 학습이 개선되는지 확인해봅니다.
# 추후에 MLP에서의 Regularization의 효과를 실험해보기 위해 
# 해당 실험에서는 regularization을 적용하지 않습니다.


```python
# -*- coding: utf-8 -*-
import json # import json module
import numpy as np
import csv
import pickle
import math
import codecs
import copy
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from NLP_MLP_functions import next_batch,val_train_set
from keras import backend as K
from keras import optimizers
from NLP_keras_evaluation import *
```

    Using TensorFlow backend.



```python
%load_ext autoreload
%autoreload 2
```


```python
with open('train_labels_with_v.pickle', 'rb') as f:
    train_labels = pickle.load(f)
with open('train_data_with_v.pickle', 'rb') as f:
    train_data = pickle.load(f)
with open('test_labels_with_v.pickle', 'rb') as f:
    test_labels = pickle.load(f)
with open('test_data_with_v.pickle', 'rb') as f:
    test_data = pickle.load(f)
```

# 두개의 층을 추가하였습니다.

- 각 층의 node 개수는 5,000개와 200개 입니다.


```python
layer_nodes=np.array([train_data.shape[1],10000,5000,200])
```


```python
from keras.layers import BatchNormalization
model = Sequential()
for idx in range(1,len(layer_nodes)):
    model.add(Dense(layer_nodes[idx], input_dim=layer_nodes[idx-1])) 
    model.add(Activation('relu'))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=[
                  recall_th(0),precision_th(0),class_F1_th(0),
                  recall_th(1),precision_th(1),class_F1_th(1),
                  recall_th(2),precision_th(2),class_F1_th(2),
                  recall_th(3),precision_th(3),class_F1_th(3),
                  recall_th(4),precision_th(4),class_F1_th(4),
                  macro_avg_recall_th,macro_avg_prec_th,macro_avg_F1_th
                      ])
```

    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:3295: The name tf.log is deprecated. Please use tf.math.log instead.
    
    Recall function is modified by TH
    Precision function is modified by TH


# 그 이후의 코드는 전 실험과 동일합니다.

# 모델 평가는 4_NLP_MLP_eval에서 이어집니다.


```python

```
