```
!pip install --upgrade pip
!pip uninstall tensorflow
!pip install tensorflow
!pip uninstall keras
!pip install keras
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
from tqdm.auto import tqdm
from glob import glob
import time, gc
#딥러닝을 사용할 때 텐서플로우, 파이토치, 케라스 등등을 사용할 수 있음. 여기서는 텐서플로우에서 케라스를 import하는데 이는
#케라스가 텐서플로우보다 상위 언어라서 그럼. 비유하자면 텐서플로우는 c언어같은 거고 케라스는 파이썬. 케라스가 더 쉽고 직관적.
from tensorflow import keras
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

import cv2
import os
import time, gc
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Input
from keras.layers import Dense, Lambda
from math import ceil

from keras.optimizers import RMSprop

# Install EfficientNet
! pip install -U git+https://github.com/qubvel/efficientnet
import efficientnet.keras as efn    
#안중요
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
```


```
#batch랑 epoch설정
batch_size = 100
epochs = 10
IMG_SIZE = 64
#색이 있는 이미지는 모두 채널 수가 3임. 흑백은 채널수가 1이고 색있는 이미지는 rgb컬러가 있어서 3임
N_CHANNELS = 3
```


```
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
```


```
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
```


```
def MNIST_reshape(x_train,y_train=None,IMG_SIZE=600):
    x_train_resized = np.zeros([0, IMG_SIZE, IMG_SIZE,3], dtype=np.float64)
    for images in x_train:
        temp = cv2.resize(images,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
        temp = cv2.cvtColor(temp,cv2.COLOR_GRAY2RGB)
        temp=temp/255
        x_train_resized=np.append(x_train_resized,np.expand_dims(temp,axis=0),axis=0)
        del temp
        del images
    del x_train
    if y_train is not None:
        y_train_resized=pd.get_dummies(y_train).values
        del y_train
        return x_train_resized,y_train_resized    
    else:
        return x_train_resized
```


```
def generator(x_train,y_train,batch_size=100,image_processing=MNIST_reshape,img_size=IMG_SIZE):
    while True:
        index=np.random.randint(0,high=x_train.shape[0],size=batch_size)
        batch_x, batch_y = image_processing(x_train[index],y_train[index],img_size)
        yield batch_x,batch_y
```


```
# inputs = Input(shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS))
```


```
import keras.backend.tensorflow_backend as K
with K.tf.device('/gpu:0'):
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS))
    model = efn.EfficientNetB7(input_tensor=inputs, weights='imagenet', include_top = False)
    #모델의 구조를 짜는 거임 
    x = model.output
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dropout(rate= 0.5, name = 'top_dropout')(x) #EfficientNet-B7에는 0.5를 적용
    Mnist_class = Dense(10,name='Mnist_class',activation='softmax')(x)

    model = Model(inputs=inputs, outputs=[Mnist_class])

    #모델을 어떻게 최적화할 것인가 설정
    model.compile(loss="categorical_crossentropy",
    optimizer=RMSprop(lr=2e-5),
    metrics=["acc"],
    )
    history=model.fit_generator(generator(x_train,y_train,batch_size=100),steps_per_epoch=x_train.shape[0]//batch_size,epochs=epochs)
```


```
print(keras.__version__)
```


```
with K.tf.device('/gpu:0'):
    history=model.fit_generator(generator(x_train,y_train,batch_size=100),steps_per_epoch=x_train.shape[0]//batch_size,epochs=epochs)
```


```

```
