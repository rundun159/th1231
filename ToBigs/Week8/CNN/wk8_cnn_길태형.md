# AlexNet

- [paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)

- [imagenet data(2012)](http://image-net.org/challenges/LSVRC/2012/index#task)

- [code](https://pytorch.org/docs/0.4.0/_modules/torchvision/models/alexnet.html)

- Model architecture
![model_architecture](https://cv-tricks.com/wp-content/uploads/2017/03/xalexnet_small-1.png.pagespeed.ic.u_mv-jhXMI.webp)

### Naive Version
CONV_1 - POOL_1 - CONV_2 - POOL_2 - CONV_3 - CONV_4 - CONV_5 - POOL_3 - FC1 - FC2 - FC3 (->SOFTMAX)

### detailed
CONV_1(ReLU) - POOL_1 - CONV_2(ReLU) - POOL_2 - CONV_3(ReLU) - CONV_4(ReLU) - CONV_5(ReLU) - POOL_3 -(Flatten) FC1(ReLU) - FC2(ReLU) - FC3(->SOFTMAX)


```python
import warnings
warnings.filterwarnings('ignore')
import keras
from keras import layers
from keras import models
```

    Using TensorFlow backend.



```python
model = models.Sequential()
```

    WARNING: Logging before flag parsing goes to stderr.
    W0907 18:50:18.480657 12672 deprecation_wrapper.py:119] From C:\Users\MyCOM\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    



```python
#최종 분류할 class의 갯수. 예를 들어 class 갯수가 10개!
Num_classes = 10
```

### Layer 1 is a Convolution Layer_1

- **Input Image size**     224 x 224 x 3 -> 227 x 227 x 3

- **Number of filters**   96

- **Filter size** 11 x 11 x 3

- **Stride** 4

- **Layer 1 Output**  55 x 55 x 96 (because of stride 4)





```python
## TODO ##

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

Conv_1 = (11*11*3) * 96 + 96 

'''last 96 for Bias'''
```




    'last 96 for Bias'




```python
#zero padding을 모든 테두리에 하는게 아니라 위쪽이랑 오른쪽에만 할 수도 있음
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.Conv2D(96, (11, 11),activation='relu', input_shape=(227, 227, 3), strides = (4,4)))
```

    W0907 18:50:18.638880 12672 deprecation_wrapper.py:119] From C:\Users\MyCOM\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0907 18:50:18.653265 12672 deprecation_wrapper.py:119] From C:\Users\MyCOM\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    



```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    =================================================================
    Total params: 34,944
    Trainable params: 34,944
    Non-trainable params: 0
    _________________________________________________________________


### Layer 2 is a Max Pooling_1 Followed by Convolution_1

- **Input**  55 x 55 x 96

- **Max pooling**  

- **Pooling size**(overlapping) 3 x 3

- **Stride** 2

- **Layer 2 Output** 27 x 27 x 96



```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

Max_pool_1 = 0
#max pooling은
#입력 데이터위에 window를 씌워서 이동시킬 때
#window에 보이는 값중 가장 큰 값을 추출하는 pooling입니다.
#따라서, 필요한 parameter는 없습니다.
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.MaxPooling2D((3, 3),strides=(2,2)))
```

    W0907 18:50:18.749490 12672 deprecation_wrapper.py:119] From C:\Users\MyCOM\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    



```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    =================================================================
    Total params: 34,944
    Trainable params: 34,944
    Non-trainable params: 0
    _________________________________________________________________


### Layer 3 is a a Convolution Layer_2

- **Input**  27 x 27 x 96

- **Number of filters**  256

- **Filter size**  5 x 5 x 96 

- **Stride** 1

- **padding** 2

- **Layer 3 Output** 

- **output의 width, height의 크기 : (27+4-5)/1 + 1 = 27**

- **Layer 3 Output Size  27 x 27 x 256(# of filters) **


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

Conv_2 = ((5*5*96)+1)*256 #= 614656
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.Conv2D(256, (5, 5),activation='relu', input_shape=(27, 27, 96), strides = (1,1),padding='SAME'))
#padding='SAME' => input의 width/height size와 output의 width/height size가 같도록 padding을 설정
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    =================================================================
    Total params: 649,600
    Trainable params: 649,600
    Non-trainable params: 0
    _________________________________________________________________


### Layer 4 is a Max Pooling_2 Followed by Convolution_2

- **Input**  27 x 27 x 256 (same as Layer 3 Output size)

- **Max pooling**  

- **Pooling size**(overlapping) 3 x 3

- **Stride** 2

- **Layer 4 Output**  13 x 13 x 256


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

Max_pool_2 = 0
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.MaxPooling2D((3, 3),strides=(2,2)))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    =================================================================
    Total params: 649,600
    Trainable params: 649,600
    Non-trainable params: 0
    _________________________________________________________________


### Layer 5 is a a Convolution Layer_3

- **Input**  13 x 13 x 256

- **Number of filters**  384

- **Filter size**  3 x 3 x 256 (same as input's 3rd channel size)

- **Stride** 1

- **padding** 1
- Alex Net 그림에 따르면, 입력층과 출력층의 size가 같으므로, 
- 이 layer에서 input/output의 size가 같게 하려면 padding의 크기가 1이어야 한다
- **Layer 5 Output** 13 x 13 x 384


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################
Conv_3 = (3*3*256 +1)*384
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.Conv2D(384, (3, 3),activation='relu', input_shape=(13, 13, 256), strides = (1,1), padding = 'SAME'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    =================================================================
    Total params: 1,534,720
    Trainable params: 1,534,720
    Non-trainable params: 0
    _________________________________________________________________


### Layer 6 is  a Convolution Layer_4

- **Input**  13 x 13 x 384

- **Number of filters**  384

- **Filter size**  3 x 3 x 384

- **Stride** 1

- **padding** 1

- **Layer 6 Output** 13 x 13 x 384


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################
Conv_4 =(3*3*384+1)*384
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.Conv2D(384, (3, 3),activation='relu', input_shape=(13, 13, 384), strides = (1,1), padding = 'SAME'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    =================================================================
    Total params: 2,862,208
    Trainable params: 2,862,208
    Non-trainable params: 0
    _________________________________________________________________


### Layer 7 is a Convolution Layer_5

- **Input**  13 x 13 x 384

- **Number of filters**  256

- **Filter size**  3 x 3 x 256

- **Stride** 1

- **padding** 1

- **Layer 7 Output** 13 x 13 x 256


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################
Conv_5 = (3*3*256+1)*256
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.Conv2D(256, (3, 3),activation='relu', input_shape=(13, 13, 384), strides = (1,1), padding = 'SAME'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
    =================================================================
    Total params: 3,747,200
    Trainable params: 3,747,200
    Non-trainable params: 0
    _________________________________________________________________


### Layer 8 is a Max Pooling_3 Followed by Convolution_5

- **Input**  13 x 13 x 256

- **Max pooling**  

- **Pooling size**(overlapping) 3 x 3

- **Stride** 2

- **Layer 8 Output** 6 x 6 x256


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

Max_pool_3 = 0
```


```python
#input과 output size에 맞는 계수 대입. 
model.add(layers.MaxPooling2D((3, 3),strides=(2,2)))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
    =================================================================
    Total params: 3,747,200
    Trainable params: 3,747,200
    Non-trainable params: 0
    _________________________________________________________________


### Layer 9 is a Fully_Connected layer_1

- **input** 6 x 6 x 256

- **flatten** 9216

- **output size** (N,flatten) x (flatten,4096)

- **N** Number of input data


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################
FC1 = 9216*4096
```


```python
model.add(layers.Flatten())
# data를 1차원으로 flat시키는 함수를 적용합니다.
```


```python
#FC 계층의 출력층의 node 갯수(4096)를 parameter로 대입합니다.
model.add(layers.Dense(4096, activation='relu'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              37752832  
    =================================================================
    Total params: 41,500,032
    Trainable params: 41,500,032
    Non-trainable params: 0
    _________________________________________________________________


### Layer 10 is a Fully_Connected layer_2

- **input** (N,4096)

- **output size** (N,4096) x (4096,4096)

- **N** Number of input data


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

FC2 = 4096*4096
```


```python
#FC 계층의 출력층의 node 갯수(4096)를 parameter로 대입합니다.
model.add(layers.Dense(4096, activation='relu'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              37752832  
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    =================================================================
    Total params: 58,281,344
    Trainable params: 58,281,344
    Non-trainable params: 0
    _________________________________________________________________


### Layer 11 is a Fully_Connected layer_3

- **input** (N,4096)

- **output size** (N,4096) x (4096,Num_classes)

- **N** Number of input data

- **Num_classes** Number of labels


```python
## TODO 

######################################################
#  Calculate the number of parameters in this layer  #
######################################################

FC3 = 4096* Num_classes
```


```python
model.add(layers.Dense(Num_classes, activation='softmax'))
#마지막 FC 계층이므로 softmax 적용
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              37752832  
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                40970     
    =================================================================
    Total params: 58,322,314
    Trainable params: 58,322,314
    Non-trainable params: 0
    _________________________________________________________________

