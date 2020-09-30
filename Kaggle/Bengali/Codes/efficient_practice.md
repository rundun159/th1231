```python
#필요한 라이브러리들을 불러오는 과정임 
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


```python
#train등등의 데이터를 불러옴. 안중요함.
#TPU에 맞게 불러옴
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path('bengaliai-cv19')
!gsutil ls $GCS_DS_PATH
!pip install gcsfs
train_df_ = pd.read_csv('gs://kds-06929b980722279141ec67c5f76a6468a973cd72023b750636bda6c7/train.csv')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
```


```python
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
```


```python
AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
```


```python
#이건 중요한데 지금 데이터가 픽셀단위로 잘려서 32,332(137x236)픽셀이 일렬로 저장되어있음. e.g. 0,0,0,0.5,0,0,0,.... 이걸 행렬로 다시
#변환하는 거임(Reshape) 그럼 137x236인 직사각형이 될 것.
HEIGHT = 137
WIDTH = 236
#이따가 resize함수가 나오는데 우리가 cnn모델의 input에 데이터를 넣을 때 137x236을 넣으면 처리해야하는 픽셀 수가 많아서 오래 걸림.
#그래서 64x64로 넣어주는거임
IMG_SIZE = 64
#색이 있는 이미지는 모두 채널 수가 3임. 흑백은 채널수가 1이고 색있는 이미지는 rgb컬러가 있어서 3임
N_CHANNELS = 3
```


```python
#CNN모델에서 input사이즈를 어떻게 넣을까에 관한 거임. 우리는 64x64x3으로 넣을거임
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 3))

#efficientNetB7모델을 넣을거임. weight는 imagenet에서 pretrain함. include_top을 False로 하면 마지막 fully connected층은 빼고 모델을 저장함
#이 fc layer를 우리는 168 + 11 + 7개 넣어줄거임.

with tpu_strategy.scope():
    model = efn.EfficientNetB7(input_tensor=inputs, weights='imagenet', include_top = False)
    #모델의 구조를 짜는 거임 
    x = model.output
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dropout(rate= 0.5, name = 'top_dropout')(x) #EfficientNet-B7에는 0.5를 적용
    head_root = Dense(168, name = 'head_root', activation = 'softmax')(x)
    head_vowel = Dense(11, name = 'head_vowel', activation = 'softmax')(x)
    head_consonant = Dense(7, name = 'head_consonant', activation = 'softmax')(x)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    #모델을 어떻게 최적화할 것인가 설정
    model.compile(loss="categorical_crossentropy",
        optimizer=RMSprop(lr=2e-5),
        metrics=["acc"],
    )
```


```python
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
learning_rate_reduction_root = ReduceLROnPlateau(monitor='head_root_acc', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='head_vowel_acc', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='head_consonant_acc', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
```


```python
#batch랑 epoch설정
batch_size = 128
epochs = 10
```


```python
# helper for mixup
def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    return r_x, r_y, r_l, r_w, r_h

# custom image data generator
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    # custom image generator
    def __init__(self, featurewise_center = False, samplewise_center = False, 
                 featurewise_std_normalization = False, samplewise_std_normalization = False, 
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0, 
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0, 
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False, 
                 vertical_flip = False, rescale = None, preprocessing_function = None, data_format = None, validation_split = 0.0, 
                 mix_up_alpha = 0.0, cutmix_alpha = 0.0): # additional class argument
    
        # parent's constructor
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, 
                         zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, 
                         shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, 
                         preprocessing_function, data_format, validation_split)

        # Mix-up
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        
        # Cutmix
        assert cutmix_alpha >= 0.0
        self.cutmix_alpha = cutmix_alpha

    def mix_up(self, X1, y1, X2, y2, ordered_outputs, target_lengths):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        target_dict = {}
        i = 0
        for output in ordered_outputs:
            target_length = target_lengths[output]
            target_dict[output] = y1[:, i: i + target_length] * y_l + y2[:, i: i + target_length] * (1 - y_l)
            i += target_length
        y = None
        for output, target in target_dict.items():
            if y is None:
                y = target
            else:
                y = np.concatenate((y, target), axis=1)
        return X, y
    
    def cutmix(self, X1, y1, X2, y2, ordered_outputs, target_lengths):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[1]
        height = X1.shape[0]
        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        target_dict = {}
        i = 0
        for output in ordered_outputs:
            target_length = target_lengths[output]
            target_dict[output] = y1[:, i: i + target_length] * lam + y2[:, i: i + target_length] * (1 - lam)
            i += target_length
        y = None
        for output, target in target_dict.items():
            if y is None:
                y = target
            else:
                y = np.concatenate((y, target), axis=1)
        return X, y
    
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        
        # for multi-outputs
        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)
        
        # parent flow
        batches = super().flow(x, targets, batch_size, shuffle, sample_weight, seed, save_to_dir, save_prefix, save_format, subset)
        
        # custom processing
        while True:
            batch_x, batch_y = next(batches)
            
            # mixup or cutmix
            if (self.mix_up_alpha > 0) & (self.cutmix_alpha > 0):
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                if np.random.rand() < 0.5:
                    batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2, ordered_outputs, target_lengths)
                else:
                    batch_x, batch_y = self.cutmix(batch_x, batch_y, batch_x_2, batch_y_2, ordered_outputs, target_lengths)
            
                target_dict = {}
                i = 0
                for output in ordered_outputs:
                    target_length = target_lengths[output]
                    target_dict[output] = batch_y[:, i: i + target_length]
                    i += target_length
                    
                yield batch_x, target_dict
```


```python
#이미지 사이즈를 64로 바꿔주는 코드임. 나도 해석 안해봐서 잘 모름 
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size=64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    resized = cv2.cvtColor(np.float32(resized), cv2.COLOR_GRAY2RGB)
    return resized
```


```python
#훈련의 메인 루프
import tensorflow as tf

histories=[]
for i in range(4):
    #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
    
    #이건 이미지 처리 시 정규화과정. 통계에서 정규화랑 비슷한거라고 보면 됨.
    X_train = resize(X_train)/255
    
    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
        
    #원 핫 인코딩 형태로 Y값들을 바꿔줌.
    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
    
    #train과 test를 나눔
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
    #메모리 정리를 위해 삭제
    del train_df
    del X_train
    del Y_train_root, Y_train_vowel, Y_train_consonant

    datagen = MultiOutputDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=16,  # randomly rotate images in the range (degrees, 0 to 180, was 8)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,
        mix_up_alpha = 0.4, 
        cutmix_alpha = 0.4)
    
    history = model.fit_generator(datagen.flow(x_train, {'head_root': y_train_root, 'head_vowel': y_train_vowel, 'head_consonant': y_train_consonant}, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                              steps_per_epoch=x_train.shape[0] //batch_size, callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])
    
    histories.append(history)
    # Delete to reduce memory usage
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
```


```python
%matplotlib inline
def plot_loss(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['head_root_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['head_vowel_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['head_consonant_loss'], label='train_consonant_loss')
    
    plt.plot(np.arange(0, epoch), his.history['val_head_root_loss'], label='val_train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_head_vowel_loss'], label='val_train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_head_consonant_loss'], label='val_train_consonant_loss')
    
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['head_root_acc'], label='train_root_accuracy')
    plt.plot(np.arange(0, epoch), his.history['head_vowel_acc'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['head_consonant_acc'], label='train_consonant_accuracy')
    
    plt.plot(np.arange(0, epoch), his.history['val_head_root_acc'], label='val_root_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_head_vowel_acc'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_head_consonant_acc'], label='val_consonant_accuracy')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()
```


```python
for dataset in range(4):
    plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')
    plot_acc(histories[dataset], epochs, f'Training Dataset: {dataset}')
```


```python
from keras.models import Model
model.save('Bengali_model_epochs2.h5')
```


```python
preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}
```


```python
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder
for i in range(4):
    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 
    df_test_img.set_index('image_id', inplace=True)

    X_test = resize(df_test_img, need_progress_bar=False)/255
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    preds = model.predict(X_test)

    for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[i], axis=1)

    for k,id in enumerate(df_test_img.index.values):  
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])
    del df_test_img
    del X_test
    gc.collect()

df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)
df_sample.head()
```
