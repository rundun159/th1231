# Machine Learning, Spring 2020, Final Project

## Image Classification with Noisy Labels

1. Download ```final_project.pdf```, ```Kaggle & Colab Guide.pptx```, and ```utils.py``` from i-campus.
2. Go to [Kaggle competition page](https://www.naver.com), join Kaggle & competition, and download dataset.
3. Following guide slides, upload ```utils.py```.
4. Mount Google Drive.
5. Implement your own model and predict on test images.
6. Download and submit ```submission.csv``` to Kaggle.
7. Write a report on your project and submit on i-campus.

# INITIAL PACKAGES


```
# INITIAL PACKAGES
import os
import numpy as np
import pandas as pd

from utils import load_data, run
```

## Mount Google Drive

Assmue you made ```final_project``` directory on the root,
and data files are there.


```
from google.colab import drive
drive.mount('/content/gdrive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive



```
gdrive_root = '/content/gdrive/My Drive'
data_path = os.path.join(gdrive_root, 'final_project')
os.listdir(data_path)
```




    ['test_id.csv',
     'sample_submission.csv',
     'test_images.npy',
     'train_id_label.csv',
     'train_images.npy',
     'valid_id_label.csv',
     'valid_images.npy',
     'utils.py',
     '__pycache__',
     'Baseline_CNN.ipynb의 사본',
     'Keras_CNN.ipynb',
     'train_y_14',
     'train_x_len_0',
     'train_y_0',
     'train_x_len_2',
     'train_x_len_3',
     'train_x_len_4',
     'train_x_len_1',
     'train_y_1',
     'train_x_len_5',
     'train_y_2',
     'train_x_len_6',
     'train_y_3',
     'train_x_len_7',
     'train_y_4',
     'train_x_len_8',
     'train_y_5',
     'train_x_len_9',
     'train_x_len_10',
     'train_y_6',
     'train_x_len_11',
     'train_y_7',
     'train_x_len_12',
     'train_x_len_13',
     'train_y_8',
     'train_y_9',
     'train_y_10',
     'train_y_11',
     'split_data.ipynb',
     'train_y_12',
     'train_y_13',
     'final.h5',
     'final2.h5',
     'm_np',
     'real_train_x',
     'real_train_y',
     'real_final.h5',
     'training_keras.ipynb',
     'real_final_2005.h5',
     'real_final_2035.h5',
     'real_train_x_2005',
     'real_train_y_2005',
     'real_final_2047.h5',
     'submission.csv',
     'Answer.ipynb',
     'get_mislabled.ipynb']




```
train_data, valid_data, test_data = load_data(data_path)
```


```
print(f'train id label:\n {train_data[0].head()}')
print(f'train images shape: {train_data[1].shape}\n')
assert len(train_data[0]) == len(train_data[1])

print(f'valid id label:\n {valid_data[0].head()}')
print(f'valid images shape: {valid_data[1].shape}\n')
assert len(valid_data[0]) == len(valid_data[1])

print(f'test id:\n {test_data[0].head()}')
print(f'test images shape: {test_data[1].shape}\n')
assert len(test_data[0]) == len(test_data[1])
```

    train id label:
                id  label
    0  a1cec2874d      1
    1  ddbe361041      7
    2  910628fd4e      2
    3  171ae22c4b     10
    4  f1e68a9c42      1
    train images shape: (100000, 3, 32, 32)
    
    valid id label:
                id  label
    0  f0829f4147      7
    1  91116a7846     11
    2  88c83f1240      3
    3  a7cd83fe4f      4
    4  78b3ce0c46      1
    valid images shape: (10000, 3, 32, 32)
    
    test id:
                id
    0  edf4ea9a4b
    1  7496e3e847
    2  e0f2110942
    3  9fb87df04a
    4  0a3608ca47
    test images shape: (10000, 3, 32, 32)
    


---

# SHOW YOUR WORK
From here, import packages you need as long as they are permitted. <br>
Fill ```train_and_predict``` function with your codes. <br>
If you want, you can implement your own classes or functions within "SHOW YOUR WOKR" block. <br>
The rest of work is ours.


```
# IMPORT PACKAGES YOU NEED
import pickle
!pip install --upgrade pip
!pip uninstall tensorflow
!pip install tensorflow
!pip uninstall keras
!pip install keras
# IMPORT PACKAGES YOU NEED
from time import time
from keras.optimizers import RMSprop
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Input, LeakyReLU, Activation
import keras.backend.tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
os.chdir(gdrive_root+'/final_project')


```

    Collecting pip
    [?25l  Downloading https://files.pythonhosted.org/packages/43/84/23ed6a1796480a6f1a2d38f2802901d078266bda38388954d01d3f2e821d/pip-20.1.1-py2.py3-none-any.whl (1.5MB)
    [K     |████████████████████████████████| 1.5MB 2.8MB/s 
    [?25hInstalling collected packages: pip
      Found existing installation: pip 19.3.1
        Uninstalling pip-19.3.1:
          Successfully uninstalled pip-19.3.1
    Successfully installed pip-20.1.1
    Found existing installation: tensorflow 2.2.0
    Uninstalling tensorflow-2.2.0:
      Would remove:
        /usr/local/bin/estimator_ckpt_converter
        /usr/local/bin/saved_model_cli
        /usr/local/bin/tensorboard
        /usr/local/bin/tf_upgrade_v2
        /usr/local/bin/tflite_convert
        /usr/local/bin/toco
        /usr/local/bin/toco_from_protos
        /usr/local/lib/python3.6/dist-packages/tensorflow-2.2.0.dist-info/*
        /usr/local/lib/python3.6/dist-packages/tensorflow/*
    Proceed (y/n)? y
      Successfully uninstalled tensorflow-2.2.0
    Collecting tensorflow
      Downloading tensorflow-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl (516.2 MB)
    [K     |████████████████████████████████| 516.2 MB 19 kB/s 
    [?25hRequirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.0)
    Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.9.0)
    Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.10.0)
    Requirement already satisfied: scipy==1.4.1; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.4.1)
    Requirement already satisfied: numpy<2.0,>=1.16.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.18.5)
    Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.29.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: tensorboard<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.2)
    Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.1)
    Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.2.1)
    Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.3.3)
    Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.34.2)
    Requirement already satisfied: keras-preprocessing>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: tensorflow-estimator<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.0)
    Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.10.0)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (2.23.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.0.post3)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.1)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (3.2.2)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (47.3.1)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.0.1)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.17.2)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2.9)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2020.4.5.2)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.0.4)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.3.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.1.0)
    Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.6)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.6/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.8)
    Installing collected packages: tensorflow
    Successfully installed tensorflow-2.2.0
    Found existing installation: Keras 2.3.1
    Uninstalling Keras-2.3.1:
      Would remove:
        /usr/local/lib/python3.6/dist-packages/Keras-2.3.1.dist-info/*
        /usr/local/lib/python3.6/dist-packages/docs/*
        /usr/local/lib/python3.6/dist-packages/keras/*
      Would not remove (might be manually added):
        /usr/local/lib/python3.6/dist-packages/docs/md_autogen.py
        /usr/local/lib/python3.6/dist-packages/docs/update_docs.py
    Proceed (y/n)? y
      Successfully uninstalled Keras-2.3.1
    Collecting keras
      Downloading Keras-2.4.2-py2.py3-none-any.whl (170 kB)
    [K     |████████████████████████████████| 170 kB 2.9 MB/s 
    [?25hRequirement already satisfied: scipy>=0.14 in /usr/local/lib/python3.6/dist-packages (from keras) (1.4.1)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from keras) (3.13)
    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from keras) (2.10.0)
    Requirement already satisfied: numpy>=1.9.1 in /usr/local/lib/python3.6/dist-packages (from keras) (1.18.5)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from h5py->keras) (1.12.0)
    Installing collected packages: keras
    Successfully installed keras-2.4.2


    Using TensorFlow backend.



```
# YOUR OWN CLASSES OR FUNCTIONS
REAL_CLASSES=14
def cnn_model(input_shape):
    with K.tf.device('/gpu:0'):
        model = Sequential()
        model.add(Conv2D(128, kernel_size=3,strides=1,padding='same',input_shape=input_shape,kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv2D(512, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(REAL_CLASSES,name='logit'))
        model.add(Activation('softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=RMSprop(lr=2e-5),
                    metrics=['accuracy'])
    return model
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
```


```
def train_and_predict(train_data, valid_data, test_data):
    """Train a model and return prediction on test images.

    Given train and valid data, build your model and optimize.
    Then, return predictions on test_images.

    You can import packages you want inside 'EDIT HERE' as long as they are permitted.
    (See document for the list of possible packages)

    arguments:
        train_data: tuple of (pandas.DataFrame, np.array).
        - 0: pandas.DataFrame with columns ['id', 'label']
          'id' contains unique id assigned to each image.
          'label' contains label (0 ~ # classes-1) corresponding to its image.
        - 1: train image in np.array of (# train data, # channel, height, width)

        valid_data: tuple of (pandas.DataFrame, np.array).
        - 0: pandas.DataFrame with columns ['id', 'label']
          'id' contains unique id assigned to each image.
          'label' contains label (0 ~ # classes-1) corresponding to its image.
        - 1: valid image in np.array of (# valid data, # channel, height, width)

        test_data: tuple of (pandas.DataFrame, np.array).
        - 0: pandas.DataFrame with columns ['id']
          'id' contains unique id assigned to each image.
        - 1: test image in np.array of (# test data, # channel, height, width)
    
    returns:
        pandas.DataFrame, predictions on test images with columns ['id', 'label'].
        'id' should contain unique id assigned to test images. 
        'label' should contain prediction on the test image correspond to its id

    """
    # Example code:
    train_id_label, train_images = train_data
    valid_id_label, valid_images = valid_data
    test_id, test_images = test_data

    num_train = len(train_images)
    num_valid = len(valid_images)
    num_test = len(test_images)

    # BUILD YOUR MODEL
    # Example: Random prediction

    train_data, valid_data, test_data = load_data(data_path)
    np_train_data=np.asarray(train_data)
    REAL_CLASSES = 14
    train_x=np.asarray(train_data[1])
    train_x=np.rollaxis(train_x, 1, 4)
    train_y=np.asarray(train_data[0]['label'])
    REAL_CLASSES = 14
    train_y_one_hot=np.zeros(shape=(len(train_y),REAL_CLASSES))
    for i in range(len(train_y_one_hot)):
        train_y_one_hot[i][train_y[i]]=1
    
    model=cnn_model((32,32,3))
    
    with K.tf.device('/gpu:0'):
        hist=model.fit(train_x, train_y_one_hot, batch_size=256, epochs=40, verbose=1)
    m_np=np.zeros(shape=len(train_y_one_hot))
    part_size=1000
    for idx in range(int(len(train_y_one_hot)/part_size)):
        print(int(len(train_y_one_hot)/part_size))
        print(idx)
        start_idx=idx*part_size
        end_idx=(idx+1)*part_size
        with K.tf.device('/gpu:0'):
            extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])    
            features = extractor(train_x[start_idx:end_idx])
        for i in range(part_size):
            m_np[i+start_idx]+=features[-2][i][np.argmax(train_y_one_hot[i+start_idx])]
            max_val=-987654321
            max_idx=-1
            for other in range(14):
                if other==np.argmax(train_y_one_hot[i+start_idx]):
                    continue
                if features[-2][i][other]>max_val:
                    max_val=features[-2][i][other]
                    max_idx=other
            m_np[i+start_idx]-=max_val
        del extractor,features
    del model
    model=cnn_model((32,32,3))
    PERCENT=0.7
    real_idx=np.argsort(m_np)[-int(len(m_np)*PERCENT):]
    real_train_x=train_x[real_idx]
    real_train_y=train_y_one_hot[real_idx]
    datagen.fit(real_train_x)
    with K.tf.device('/gpu:0'):
        model.fit(datagen.flow(real_train_x, real_train_y, batch_size=64),
                    steps_per_epoch=len(real_train_x) / 64, epochs=60,validation_data=(val_x,val_np_y))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=RMSprop(lr=5e-6),
                metrics=['accuracy'])
    with K.tf.device('/gpu:0'):
        model.fit(datagen.flow(real_train_x, real_train_y, batch_size=64),
                    steps_per_epoch=len(real_train_x) / 64, epochs=100,validation_data=(val_x,val_np_y))    
    test_x=np.rollaxis(test_x, 1, 4)
    result=model.predict(test_x)
    result_idx=np.zeros(len(result),dtype=np.int)
    for i in range(len(result)):
        result_idx[i]=np.argmax(result[i])
    # Make prediction data frame
    test_id['label'] = result_idx
    pred = test_id.loc[:, ['id', 'label']]
    
    return pred
```

---

# YOUR WORK IS DONE!
Do not touch any line below. <br>
```run``` function will grap your prediction and make ```submission.csv```. <br>
Take it and submit to Kaggle!


```
run(train_and_predict, train_data, valid_data, test_data)
```

    Epoch 1/40
    391/391 [==============================] - 55s 141ms/step - loss: 2.5156 - accuracy: 0.1974
    Epoch 2/40
    391/391 [==============================] - 55s 141ms/step - loss: 2.2460 - accuracy: 0.2792
    Epoch 3/40
    391/391 [==============================] - 55s 141ms/step - loss: 2.1167 - accuracy: 0.3209
    Epoch 4/40
    391/391 [==============================] - 55s 141ms/step - loss: 2.0237 - accuracy: 0.3486
    Epoch 5/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.9513 - accuracy: 0.3715
    Epoch 6/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.8932 - accuracy: 0.3908
    Epoch 7/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.8437 - accuracy: 0.4072
    Epoch 8/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.7958 - accuracy: 0.4229
    Epoch 9/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.7511 - accuracy: 0.4363
    Epoch 10/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.7124 - accuracy: 0.4505
    Epoch 11/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.6759 - accuracy: 0.4598
    Epoch 12/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.6432 - accuracy: 0.4717
    Epoch 13/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.6110 - accuracy: 0.4837
    Epoch 14/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.5819 - accuracy: 0.4921
    Epoch 15/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.5541 - accuracy: 0.5006
    Epoch 16/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.5272 - accuracy: 0.5094
    Epoch 17/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.5001 - accuracy: 0.5184
    Epoch 18/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.4734 - accuracy: 0.5279
    Epoch 19/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.4501 - accuracy: 0.5359
    Epoch 20/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.4314 - accuracy: 0.5389
    Epoch 21/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.4060 - accuracy: 0.5477
    Epoch 22/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.3856 - accuracy: 0.5568
    Epoch 23/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.3627 - accuracy: 0.5627
    Epoch 24/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.3393 - accuracy: 0.5703
    Epoch 25/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.3193 - accuracy: 0.5769
    Epoch 26/40
    391/391 [==============================] - 55s 142ms/step - loss: 1.2996 - accuracy: 0.5829
    Epoch 27/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.2821 - accuracy: 0.5902
    Epoch 28/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.2642 - accuracy: 0.5936
    Epoch 29/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.2436 - accuracy: 0.6015
    Epoch 30/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.2255 - accuracy: 0.6066
    Epoch 31/40
    391/391 [==============================] - 55s 141ms/step - loss: 1.2052 - accuracy: 0.6116
    Epoch 32/40
    383/391 [============================>.] - ETA: 1s - loss: 1.1895 - accuracy: 0.6184


```

```
