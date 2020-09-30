```
# for i in range(4):
#     print(i)
```


```
# function ConnectButton(){
# console.log("Working"); 
# document.querySelector("#connect").click() 
# }
# setInterval(ConnectButton,60000)

# function ClickConnect() {  
#     var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); 
#     buttons.forEach(function(btn) { btn.click(); }); 
#     console.log("1분마다 자동 재연결"); 
#     document.querySelector("#top-toolbar > colab-connect-button").click(); 
# }
# setInterval(ClickConnect,1000*60);

```


```
!pip install --upgrade pip
!pip uninstall tensorflow
!pip install tensorflow
!pip uninstall keras
!pip install keras

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


```
#google drive를 google colab이랑 연동하기. 
#아래 뜨는 링크 눌러서 권한을 받아와야함.
#authorization link를 복붙하면 됨
from google.colab import drive

drive.mount('/content/gdrive')
root_path = '/content/gdrive/My Drive/kaggle/Bengali'  #change dir to your project folder
```


```
import os
os.chdir(root_path+'/input')  #change dir
```


```
!ls
```


```
train_df_ = pd.read_csv('train/train.csv')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
test_df_ = pd.read_csv('test/test.csv')
sample_sub_df = pd.read_csv('sample_submission.csv')
```


```
train_df_
```


```
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


```
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 3))
```


```
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
#batch랑 epoch설정
batch_size = 32
epochs = 3

```


```
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


```
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


```
!ls model_saved
```


```
import keras.backend.tensorflow_backend as K
with K.tf.device('/gpu:0'):
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS))
    model = efn.EfficientNetB4(input_tensor=inputs, weights='imagenet', include_top = False)
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
!ls train
!ls
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    count=1
    while True:
        print(count)
        for i in range(4):
            #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
            train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
            model.save('model_saved/efficient'+str(i)+'.h5')
        count=count+1
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```
#훈련의 메인 루프
import tensorflow as tf

with K.tf.device('/gpu:0'):
    histories=[]
    for i in range(4):
        #train_df에 픽셀값들만 남기고 나머지 열들을 없앰
        train_df = pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
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
        model.save('model_saved/efficient'+str(i)+'.h5')
        print(i)
```


```

```
