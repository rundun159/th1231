### Matrix Shape 정리
$ X == (N,D) $  
$ W_1 == (D,H) $  
$ b_1 == (H,) $  
$ H == (N,H) $  
$ A == (N,H) $  
$ W_2 == (H,C) $  
$ b_2 == (C,) $  
$ S == (N,C) $  
$ P == (N,C) $  

### Matrix 미분 정리  
$ H = XW + b\qquad  (N, H) = (N, D) \times (D, H) + (H,) $  
$ L = f(H) $  
$ {\partial L \over \partial W} = X^T {\partial L \over \partial H} = {\partial H \over \partial W} \times {\partial L \over \partial H}$  
$ {\partial L \over \partial X} = {\partial L \over \partial H} W^T = {\partial L \over \partial H} \times {\partial H \over \partial X}$  
$ {\partial L \over \partial b} = 1*{\partial L \over \partial H}  $  

### 2 Layers Chain Rule 정리  
##### Forward
$ H = XW_1 + b_1$  
$ A = ReLU(H) $  
$ S = AW_2 + b_2 $  
$ P = Softmax(S) $    
$ L = -LogLikelihood(P) $  
##### Backward  
$ {\partial L \over \partial S} = ? $ : T는 Label  
$ {\partial L \over \partial W_2} = {\partial S \over \partial W_2}{\partial L \over \partial S} = ? $  
$ {\partial L \over \partial b_2} = 1 * {\partial L \over \partial S} = ? $  
$ {\partial L \over \partial A} = {\partial L \over \partial S}{\partial S \over \partial A} = ? $  
$ {\partial L \over \partial H} = ? * {\partial L \over \partial A} $  
$ {\partial L \over \partial W_1} = {\partial H \over \partial W_1}{\partial L \over \partial H} = ? {\partial L \over \partial H}  $  
$ {\partial L \over \partial b_1} = ? $  

### Softmax - Cross Entropy Error미분  
<img src="img/fig a-5.png">

### 데이터 Load  
cifar-10 데이터를 불러옵니다.  
프레임워크 내 자체적으로 데이터를 로드할 수 있지만, 
이렇게도 데이터 로드가 가능합니다!  
32  32  3 차원의 데이터를 3072 차원으로 바뀌는 것 까지 드릴게요.


```python
from Model import TwoLayerNet
```


```python
from load_cifar_10 import *
import numpy as np
from Model import TwoLayerNet
```


```python
cifar_10_dir = 'cifar-10-batches-py'
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)
```


```python
def Processing_data(train, test):
    #change dtype
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    
    #Reshaping
    train = np.reshape(train, (train.shape[0], -1))
    test = np.reshape(test, (test.shape[0], -1))
    
    #Normalizing
    mean_image = np.mean(train, axis = 0)
    #print(train.dtype)
    train -= mean_image
    test -= mean_image
    
    return train, test
```


```python
train_data, test_data = Processing_data(train_data, test_data)
```


```python
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
```

    (50000, 3072)
    (50000,)
    (10000, 3072)
    (10000,)


너무 많으니까 5000개, 1000개만 사용합시다!


```python
train_data = train_data[:5000]
train_labels = train_labels[:5000]
test_data = test_data[:1000]
test_labels = test_labels[:1000]
```

### 데이터 확인  
실제 데이터가 어떻게 생겼는지는 한번 확인해보세요!


```python
train_data
```




    array([[ -71.71074   ,  -74.05614   ,  -69.5538    , ...,   -3.6390762 ,
             -33.850304  ,  -42.38186   ],
           [  23.28926   ,   40.943863  ,   54.446198  , ...,   16.360924  ,
               7.1496964 ,   29.618141  ],
           [ 124.28926   ,  118.94386   ,  122.4462    , ...,  -46.639076  ,
             -39.850304  ,  -30.381859  ],
           ...,
           [  36.28926   ,   26.943863  ,   12.4461975 , ...,  -84.63908   ,
             -47.850304  ,  -30.381859  ],
           [  23.28926   ,   15.943863  ,   -7.5538025 , ...,   67.36092   ,
             121.1497    ,   -0.38185883],
           [ -85.71074   , -104.05614   , -111.5538    , ...,   29.360924  ,
              16.149696  ,  -14.381859  ]], dtype=float32)



### 하이퍼파라미터 설정  
하이퍼파라미터를 설정하겠습니다.  
hidden_size, epoch_size, batch_size, learning_rate 등은 전부 하이퍼 파라미터이니 바꿔서 해보세요.


```python
input_size = 32 * 32 * 3
hidden_size = 50
output_size = 10
epoch_size = 100
batch_size = 100
learning_rate = 0.0001
N = train_data.shape[0]
```

### 모델 만들기  
input_size, hidden_size, output_size는 데이터에 맞게 잘 설정해주세요.  
Model.py를 완성시켜주세요.


```python
nn = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
```


```python
nn.params['W1'].shape
```




    (3072, 50)




```python
batch_mask = np.random.choice(N, batch_size) #이번 배치에서 쓸 데이터들 인덱스 추출
x_batch = train_data[batch_mask]
t_batch = train_labels[batch_mask]

```


```python
x_batch.shape
```




    (100, 3072)




```python
t_batch.shape
```




    (100,)




```python
nn.backward(x_batch, t_batch)
```

    (100, 10)
    (10,)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-15-04b9c7635101> in <module>
    ----> 1 nn.backward(x_batch, t_batch)
    

    ~\Google 드라이브\Study\BIgData\Week6\6_NEUR~1\Model.py in backward(self, X, y, learning_rate)
        121         print(self.params['b2'].shape)
        122         self.params["W2"] -= learning_rate * grads["W2"]
    --> 123         self.params["b2"] -= learning_rate * grads["b2"]
        124         self.params["W1"] -= learning_rate * grads["W1"]
        125         self.params["b1"] -= learning_rate * grads["b1"]


    ValueError: non-broadcastable output operand with shape (10,) doesn't match the broadcast shape (100,10)



```python
history = {'val_acc': [],'val_loss': []} #기록해서 그림 그리자!

#코드를 보며 epoch, batch에 대해서 이해해봅시다.
for i in range(epoch_size):
    for j in range(N//batch_size):
        batch_mask = np.random.choice(N, batch_size) #이번 배치에서 쓸 데이터들 인덱스 추출
        x_batch = train_data[batch_mask]
        t_batch = train_labels[batch_mask]
        
        nn.backward(x_batch, t_batch) # 가중치 갱신
    
    #accuracy와 loss를 기록해둡시다.
    history["val_acc"].append(nn.accuracy(test_data, test_labels))
    history["val_loss"].append(nn.forward(test_data, test_labels))
    
    if i % 10 == 0:
        print(i, "test accuracy :", nn.accuracy(test_data, test_labels))
        print(i, "test loss     :", nn.forward(test_data, test_labels))
```


```python
a=np.array([[1,2],[3,4]])
```


```python
a.sum(axis=0)
```




    array([4, 6])



### 그림 그리기


```python
fig = plt.figure()
ax_acc = fig.add_subplot(111)

ax_acc.plot(range(epoch_size), history['val_acc'], label='정확도(%)', color='darkred')
#plt.text(3, 14.7, "<----------------정확도(%)", verticalalignment='top', horizontalalignment='right')
plt.xlabel('epochs')
plt.ylabel('Validation Accuracy(%)')
ax_acc.grid(linestyle='--', color='lavender')
ax_loss = ax_acc.twinx()
ax_loss.plot(range(epoch_size), history['val_loss'], label='오차', color='darkblue')
#plt.text(3, 2.2, "<----------------오차", verticalalignment='top', horizontalalignment='left')
plt.ylabel('Validation Error')
ax_loss.yaxis.tick_right()
ax_loss.grid(linestyle='--', color='lavender')

# 그래프 표시
plt.show()
```


```python

```
