# Regularization을 적용하면 학습이 개선되는지 실험해봅니다.


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

## 명사, 동사 모두 있는 data set을 가져옵니다.


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


```python
layer_nodes=np.array([train_data.shape[1],10000])
```

# 모델을 설계할 때, Affine 계층에 regularization을 적용했습니다.

저는 이전의 제 신경망들의 학습 패턴을 보면 
train data의 loss값은 계속 감소하지만, test data의 loss값은
5~6번째 에폭까지는 감소하다가 그 후로는 지속적으로 증가하는 것을 관찰했습니다. 

그래서 저는 이 문제는 overfitting의 문제라고 판단했고,
신경망의 overfitting을 완화하는 대표적인 방법인 regularization을 적용해야겠다고 생각했습니다. 

수업 시간에 교수님께서 오캄의 면도날(Occam's Razor)과 regularization에 대해서 설명 해주신 내용과 
여러 문서를 찾아보며 제가 스스로 생각해본 결과, 제가 이해한 Regularization의 논리는 아래와 같습니다.

Regularization의 논리에서는 모델의 복잡도를 가중치의 절대값으로 측정하는 것 같습니다. 

그리고 그것이 설득력이 있는 이유는, 
train data에서는 잘 작동했던 가중치는, input 분포가 다른 test data를 입력 받았을때, 
Affine 계층해서 선형변환을 거치므로 절대값이 클 수록 그 오차가 더욱 커지게 되고, 그 커지게 된 값이 다음 계층으로 넘어가기 때문입니다.

제가 생각하는 가중치의 절대값이나 제곱한 값을 loss function에 더해줬을때, overfitting을 막을 수 있는 이유는 이렇습니다.

어느 정도 학습이 진행되어서 train data에 대해 신경망이 일정량 학습을 했고, 
그 과정에서 가중치의 절대값이 커지게 된 상황을 가정해보겠습니다.

regularization을 적용한 신경망과, regularizaion 을 적용하지 않은 신경망의 loss function을 비교해보면 
전자의 loss function이 후자의 loss function 값보다 월등히 작을 것입니다. 

그 의미는 그만큼 학습을 진행하지 않는다는 의미 일 것입니다. 

반면 후자의 경우, 아직 loss function의 값이 크므로 loss function이 작아지는 방향으로 학습을 할텐데, 
그 방향은 가중치의 절대값이나 제곱한 값이 작아지는 방향일 것입니다.

loss function에 cross entropy 뿐만 아니라, 가중치의 절대값이나 제곱한 값을 같이 더해주므로 softmax 결과와 label 값의 차이가 작으면서, 가중치의 절대값이나 제곱한 값이 작은 지점을 절충하며 찾아갈 것입니다.

저는 모델을 설계할때, loss function에 가중치의 제곱을 더해주는 L2 regularization을 적용했습니다. 

즉, 학습을 할때, 가중치의 절대값에 더 많은 비중을 두겠다는 의미입니다. 


```python
model = Sequential()
for idx in range(1,len(layer_nodes)):
    model.add(Dense(layer_nodes[idx], input_dim=layer_nodes[idx-1],kernel_regularizer=keras.regularizers.l2(0.001))) 
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


# 그 이후로는 같은 코드이므로 생략합니다.
# 모델 평가는 3_NLP_Regularized_eval 에서 이어집니다.
