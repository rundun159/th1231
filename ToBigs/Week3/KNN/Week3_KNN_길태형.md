```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import math
```


```python
iris = load_iris()
```


```python
#iris.data -> features, iris.target -> labels
X = iris.data
y = iris.target
```

- Data 내용들
- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species _ target 변수. 어떤 종류의 iris인지. 0, 1, 2가 있음

- X_train을 봤을때, 정수형인 데이터가 없는것을 보면 Id Data는 생략된 것 같습니다. 

- About data : https://www.kaggle.com/uciml/iris


```python
# split into test and train dataset, and use random_state=48
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=35)
```


```python
from sklearn.preprocessing import StandardScaler
```

- Documentation for "StandardScaler" : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


```python
X_train[0]
```




    array([5.7, 2.5, 5. , 2. ])




```python
X_train
```




    array([[5.7, 2.5, 5. , 2. ],
           [5.5, 2.4, 3.8, 1.1],
           [4.4, 3. , 1.3, 0.2],
           [4.3, 3. , 1.1, 0.1],
           [4.6, 3.4, 1.4, 0.3],
           [6.6, 2.9, 4.6, 1.3],
           [6.1, 3. , 4.6, 1.4],
           [6.8, 3.2, 5.9, 2.3],
           [6.7, 3.1, 4.7, 1.5],
           [5. , 3.5, 1.6, 0.6],
           [5.4, 3.4, 1.5, 0.4],
           [5.4, 3.7, 1.5, 0.2],
           [6.3, 2.7, 4.9, 1.8],
           [4.8, 3.1, 1.6, 0.2],
           [7.7, 2.6, 6.9, 2.3],
           [5. , 3.4, 1.5, 0.2],
           [5.9, 3.2, 4.8, 1.8],
           [4.9, 2.5, 4.5, 1.7],
           [4.6, 3.6, 1. , 0.2],
           [6.6, 3. , 4.4, 1.4],
           [6.1, 2.6, 5.6, 1.4],
           [4.8, 3. , 1.4, 0.1],
           [7.4, 2.8, 6.1, 1.9],
           [6.4, 3.2, 5.3, 2.3],
           [6. , 2.2, 5. , 1.5],
           [5.1, 2.5, 3. , 1.1],
           [5.1, 3.8, 1.6, 0.2],
           [4.9, 3.1, 1.5, 0.2],
           [5.6, 3. , 4.1, 1.3],
           [7.2, 3.6, 6.1, 2.5],
           [6.5, 3.2, 5.1, 2. ],
           [5.2, 4.1, 1.5, 0.1],
           [5.9, 3. , 5.1, 1.8],
           [6.5, 3. , 5.5, 1.8],
           [5.5, 2.3, 4. , 1.3],
           [6.1, 3. , 4.9, 1.8],
           [5.8, 2.6, 4. , 1.2],
           [6.3, 2.3, 4.4, 1.3],
           [6.7, 3. , 5. , 1.7],
           [5.8, 4. , 1.2, 0.2],
           [5.6, 2.5, 3.9, 1.1],
           [6.4, 3.2, 4.5, 1.5],
           [5.7, 2.9, 4.2, 1.3],
           [5.1, 3.7, 1.5, 0.4],
           [5.7, 2.6, 3.5, 1. ],
           [4.4, 3.2, 1.3, 0.2],
           [6.3, 3.4, 5.6, 2.4],
           [5.2, 3.5, 1.5, 0.2],
           [6.7, 2.5, 5.8, 1.8],
           [6.4, 2.9, 4.3, 1.3],
           [6.2, 2.9, 4.3, 1.3],
           [4.8, 3.4, 1.9, 0.2],
           [4.6, 3.2, 1.4, 0.2],
           [5.6, 2.8, 4.9, 2. ],
           [6.3, 2.8, 5.1, 1.5],
           [6.7, 3.1, 4.4, 1.4],
           [5.5, 2.5, 4. , 1.3],
           [5. , 3.2, 1.2, 0.2],
           [5. , 3.6, 1.4, 0.2],
           [7. , 3.2, 4.7, 1.4],
           [4.9, 3.6, 1.4, 0.1],
           [6.1, 2.8, 4.7, 1.2],
           [4.8, 3.4, 1.6, 0.2],
           [5.1, 3.5, 1.4, 0.2],
           [5.6, 3. , 4.5, 1.5],
           [5.5, 3.5, 1.3, 0.2],
           [6. , 2.7, 5.1, 1.6],
           [7.7, 3.8, 6.7, 2.2],
           [5.4, 3.9, 1.7, 0.4],
           [5.3, 3.7, 1.5, 0.2],
           [5. , 2. , 3.5, 1. ],
           [6.3, 2.9, 5.6, 1.8],
           [6.5, 2.8, 4.6, 1.5],
           [6.1, 2.8, 4. , 1.3],
           [5.8, 2.8, 5.1, 2.4],
           [6.1, 2.9, 4.7, 1.4],
           [4.4, 2.9, 1.4, 0.2],
           [5.8, 2.7, 5.1, 1.9],
           [6.7, 3.3, 5.7, 2.1],
           [7.3, 2.9, 6.3, 1.8],
           [5.6, 2.7, 4.2, 1.3],
           [4.5, 2.3, 1.3, 0.3],
           [5.7, 2.8, 4.1, 1.3],
           [7.6, 3. , 6.6, 2.1],
           [7.1, 3. , 5.9, 2.1],
           [5. , 3.4, 1.6, 0.4],
           [6.3, 3.3, 6. , 2.5],
           [5. , 2.3, 3.3, 1. ],
           [5.2, 3.4, 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [5.1, 3.3, 1.7, 0.5],
           [5.8, 2.7, 3.9, 1.2],
           [5.4, 3.4, 1.7, 0.2],
           [6.8, 3. , 5.5, 2.1],
           [6.9, 3.1, 5.1, 2.3],
           [7.7, 2.8, 6.7, 2. ],
           [5.5, 2.4, 3.7, 1. ],
           [6.4, 3.1, 5.5, 1.8],
           [7.2, 3. , 5.8, 1.6],
           [6.4, 2.8, 5.6, 2.2],
           [6.7, 3.1, 5.6, 2.4],
           [6.2, 2.2, 4.5, 1.5],
           [5.1, 3.8, 1.9, 0.4],
           [4.7, 3.2, 1.6, 0.2],
           [5.8, 2.7, 4.1, 1. ],
           [6.2, 3.4, 5.4, 2.3],
           [6.9, 3.1, 4.9, 1.5],
           [7.7, 3. , 6.1, 2.3],
           [5.5, 2.6, 4.4, 1.2],
           [5.4, 3.9, 1.3, 0.4],
           [6. , 2.2, 4. , 1. ],
           [4.6, 3.1, 1.5, 0.2],
           [5.8, 2.7, 5.1, 1.9],
           [6. , 3.4, 4.5, 1.6],
           [5.7, 3.8, 1.7, 0.3],
           [6.7, 3. , 5.2, 2.3],
           [4.9, 3. , 1.4, 0.2],
           [7.2, 3.2, 6. , 1.8],
           [5.1, 3.4, 1.5, 0.2],
           [5.2, 2.7, 3.9, 1.4],
           [4.8, 3. , 1.4, 0.3],
           [4.9, 3.1, 1.5, 0.1],
           [6. , 3. , 4.8, 1.8],
           [5.1, 3.8, 1.5, 0.3],
           [7.9, 3.8, 6.4, 2. ],
           [6.4, 2.7, 5.3, 1.9],
           [6.8, 2.8, 4.8, 1.4],
           [6.3, 3.3, 4.7, 1.6],
           [5.7, 2.8, 4.5, 1.3],
           [6.2, 2.8, 4.8, 1.8],
           [4.9, 2.4, 3.3, 1. ],
           [6.5, 3. , 5.2, 2. ],
           [6.9, 3.1, 5.4, 2.1],
           [5.5, 4.2, 1.4, 0.2],
           [5.7, 4.4, 1.5, 0.4]])




```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
```


```python
trainSet=np.column_stack((X_train, y_train))
testSet=np.column_stack((X_test, y_test))
```

### KNN Classifier Implementation


```python
def getDistance(p, instance1, instance2):          
    # p=1 : return Manhattan Distance
    # p=2 : return Eucludean Distance

    #아래의 코드를 보니, target Data를 따로 분리하지 않는것 같아서, [0:-1]로 slicing 해서 마지막에 있는 index를 제외했습니다. 
    if p==1:
        distance = np.absolute(instance1[0:-1]-instance2[0:-1]).sum()
    else :
        distance = np.square(instance1[0:-1]-instance2[0:-1]).sum()        
    return distance
```


```python
def getNeighbors(p, trainSet, testInstance, k):
    #예외 상황  처리 : k가 trainSet의 길이보다 클때
    setLen = len(trainSet)
    if k>=setLen :
        return trainSet
 
    # p=1 : return Manhattan Distance
    # p=2 : return Eucludean Distance
    distance = np.empty(len(trainSet),dtype=np.float32) #testInstace로부터의 거리를 저장하는 변수
    neighbors_idx = np.empty(k+2,dtype=int)            #k개의 이웃들의 index. 자기 자신까지 포함시키고, 맨 뒤에 dummy까지 포함해서길이는 k+2
    neighbors_d = np.full(k+2,math.inf)       #k개의 이웃들까지의 거리. 무한대로 초기화
    neighbors = np.empty(k+2,dtype=type(testInstance))  #k개의 이웃들의 data

    for i in range(0, setLen):                   #testInstance로부터 trainSet의 각 데이터까지의 거리 구하기
        distance[i]=getDistance(p,trainSet[i],testInstance)
    for i in range(0, setLen):
        for idx in range(k,-1,-1):
            contained = False
            least = True
            if(distance[i]<=neighbors_d[idx]):
                contained = True
                neighbors_idx[idx+1]=neighbors_idx[idx]
                neighbors_d[idx+1]=neighbors_d[idx]
                neighbors[idx+1]=neighbors[idx]
            else:
                least = False
                break
        if(least):
            neighbors_idx[0]=i
            neighbors_d[0]=distance[i]
            neighbors[0]=trainSet[i]
        else:
            neighbors_idx[idx+1]=i
            neighbors_d[idx+1]=distance[i]
            neighbors[idx+1]=trainSet[i]
    return neighbors
#실제 k개의 이웃들은 index가 1부터 k이하 까지에 있는 instance들입니다.
```


```python
import operator

def getResponse(neighbors):
    total=0
    for i in neighbors:
        total += i[-1]
    vote = round((total-neighbors[0][-1]-neighbors[-1][-1])/(len(neighbors)-2))
    return int(vote)
```


```python
def getAccuracy(testSet, predictions):
    accuracy_score = 0
    count=0
    for i in testSet:
        if(i[-1]==int(predictions[count])):
            accuracy_score+=1
        count+=1
    return (accuracy_score/len(testSet))*100
```


```python
k = 3
p = 2 # Euclidean distance
```


```python
predictions=[]

for i in range(len(testSet)):
    neighbors = getNeighbors(p, trainSet, testSet[i], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print(str(i) + ' > predicted : ' + str(result) + ', actual : ' + str(testSet[i][-1]))
```

    0 > predicted : 1, actual : 1.0
    1 > predicted : 1, actual : 1.0
    2 > predicted : 2, actual : 2.0
    3 > predicted : 1, actual : 1.0
    4 > predicted : 0, actual : 0.0
    5 > predicted : 2, actual : 2.0
    6 > predicted : 2, actual : 2.0
    7 > predicted : 1, actual : 1.0
    8 > predicted : 1, actual : 1.0
    9 > predicted : 0, actual : 0.0
    10 > predicted : 1, actual : 1.0
    11 > predicted : 2, actual : 2.0
    12 > predicted : 0, actual : 0.0
    13 > predicted : 2, actual : 2.0
    14 > predicted : 0, actual : 0.0



```python
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: ' + str(accuracy) + '%')
```

    Accuracy: 100.0%


- random_state 값을 변경하고, train_set의 비율을 0.1로 낮췄더니 Accuracy가 높아졌습니다!
