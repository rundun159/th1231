# 가장 간단한 (vanilla) 신경망을 구현했습니다. 


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

## Data input

### 명사만 추출한 data를 사용합니다


```python
with open('train_labels.pickle', 'rb') as f:
    train_labels = pickle.load(f)
with open('train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)
with open('test_labels.pickle', 'rb') as f:
    test_labels = pickle.load(f)
with open('test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)
```

## 신경망 계층들의 node수를 미리 설정합니다.

- 간단하게 Hiddenl later는 하나만 만들고 학습을 진행해봤습니다.
- 15,000개의 node를 설정했습니다.


```python
layer_nodes=np.array([train_data.shape[1],15000])
```

#### Keras 라이브러리 사용법은 아래 링크를 참고했습니다.
#### 출처 : https://keras.io/getting-started/sequential-model-guide/

# Model 정의

각 계층마다 선형 변환과 Relu activation만을 진행했습니다.

각 계층의 node는 layer_nodes에 정의된 대로 구현 됩니다.(이 신경망에서는 한개의 층만 add하게 됩니다.)

optimizer는 기본적인 SGD의 발전적인 모델로 알려져있는 adagrad를 사용했습니다.

adagrad는 learning rate를 학습 상황에 맞춰서 조절할 수 있는 모델이라고 알고 있습니다.

현재 분류 문제가 다중 클래스 분류이기 때문에, 

loss function으로는 categorical_crossentropy를 사용했습니다.

모델 평가 기준으로는 recall 과 precision을 사용했습니다.

주어진 문제 상황이 binary classification이 아니라 다중 클래스 분류이기 때문에

각 클래스에 대해서 recall과 precision과 F1을 계산했고, history에 저장했으며, 

모든 클래스에 대한 metrics들을 평균내어 macro average metrics를 계산했습니다.

recall,precision, F1을 구하는 함수는 제가 직접 customize했으며, NLP_keras_evaluation.py 에서 코드를 확인하실수 있습니다.

Threshold는 0.5로 설정했습니다. 즉, 각 클래스의 score 값이 0.5보다 크면, 신경망이 해당 문서를 해당 클래스로 분류한 것이라고 판단했습니다.


```python
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
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From C:\Users\xogud\Anaconda3\envs\csSkku\lib\site-packages\keras\backend\tensorflow_backend.py:3295: The name tf.log is deprecated. Please use tf.math.log instead.
    
    Macro avg Recall_th function is ready
    Macro avg Precision_th function is ready
    Macro avg F1_th function is ready


## Model 학습

- batch size는 임의로 100으로 설정하고 진행했습니다.

학습이 잘 되는지의 여부를 확인하기 위해 test data를 validation set으로 사용했습니다.

test data가 주어지지 않았다면, K-fold valiation을 사용해야 했겠지만,

test data와 그 label이 주어졌으므로, 학습에만 참여시키지 않고 모델을 평가하는 용도로만 사용했습니다.


```python
# history_callback=model.fit(train_data, train_labels, epochs=30,batch_size=100,
#                                       validation_data=(test_data, test_labels))
```


```python
from keras.models import model_from_json 
json_file = open("./keras_model/vanilla_model.json", "r")
loaded_model_json = json_file.read() 
json_file.close()
model = model_from_json(loaded_model_json)
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

    Macro avg Recall_th function is ready
    Macro avg Precision_th function is ready
    Macro avg F1_th function is ready



```python
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=100)
```

    1497/1497 [==============================] - 14s 9ms/step


# 학습 결과 분석은 0_NLP_vailla_mode_eval에서 이어집니다.
