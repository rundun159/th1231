```python
# 데이터 로드
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from scipy import io
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
#서버 오류 -> 별도로 다운
#mnist = fetch_mldata("MNIST original")
#X = mnist.data / 255.0
#y = mnist.target

#7만개의 작은 숫자 이미지
#행 열이 반대로 되어있음 -> 전치
mnist = io.loadmat('mnist-original.mat') 
#데이터의 갯수를 10000개로 랜덤하게 뽑기.
a=np.random.choice(70000, 10000,replace=False)
X = np.array(pd.DataFrame(mnist['data'].T[a]))
y = np.array(pd.DataFrame(mnist['label'].T[a]))

# grayscale 28x28 pixel = 784 feature
# 각 picel은 0~255의 값
# label = 1~10

print (X.shape, y.shape)
```

    (10000, 784) (10000, 1)



```python
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
df['y']=0
```


```python
df['y'] = y
print('Size of the dataframe: {}'.format(df.shape))
```

    Size of the dataframe: (10000, 785)



```python
# 데이터 형태 시각화

import matplotlib.pyplot as plt

rndperm = np.random.permutation(df.shape[0])

# Plot the graph
plt.gray()
fig = plt.figure( figsize=(12,8) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'y'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_4_1.png)


## Train, Test Data Split


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
```

## PCA 

###  Data Frame 만들기


```python
df_train=pd.DataFrame(X_train)
df_train.columns=feat_cols
df_train['y']=y_train
```


```python
df_test=pd.DataFrame(X_test)
df_test.columns=feat_cols
df_test['y']=y_test
```

### n=3으로 train data이용해서 model에 fit 시키기. 

### PCA가 잘 되었는지 공간상으로 확인하기 


```python
from sklearn.decomposition import PCA

time_start = time.time()

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_train[feat_cols].values)
pca_df=pd.DataFrame()    #PCA를 통해 차원 축소한 결과를 
pca_df['pca-one'] = pca_result[:,0]
pca_df['pca-two'] = pca_result[:,1] 
pca_df['pca-three'] = pca_result[:,2] 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for i in range(10):
    ax.scatter(pca_df['pca-one'][df_train['y']==i]
               , pca_df['pca-two'][df_train['y']==i]
               , pca_df['pca-three'][df_train['y']==i]
               , s = 10)
ax.set_xlabel('PCA - ONE')
ax.set_ylabel('PCA - Two')
ax.set_zlabel('PCA - Three')
plt.show()
```

    Explained variation per principal component: [0.09827797 0.07144009 0.06202222]



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_13_1.png)


### Fit시켰던 model을 이용해서 test data transform 하기 & 3차원에 표시하기


```python
pca_df_test=pd.DataFrame()    #PCA를 통해 차원 축소한 결과를 
pca_df_test['pca-one'] = pca_test_result[:,0]
pca_df_test['pca-two'] = pca_test_result[:,1] 
pca_df_test['pca-three'] = pca_test_result[:,2] 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for i in range(10):
    ax.scatter(pca_df_test['pca-one'][df_test['y']==i]
               , pca_df_test['pca-two'][df_test['y']==i]
               , pca_df_test['pca-three'][df_test['y']==i]
               , s = 10)
ax.set_xlabel('PCA - ONE')
ax.set_ylabel('PCA - Two')
ax.set_zlabel('PCA - Three')
plt.show()
```

    Explained variation per principal component: [0.09827797 0.07144009 0.06202222]



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_15_1.png)



```python
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df_test[feat_cols].values)
time_start = time.time()
pca_test_result=pca.transform(df_test[feat_cols].values)
print( 't-SNE done! Time elapsed: {} seconds',time.time() - time_start )
```

    t-SNE done! Time elapsed: {} seconds 0.030253887176513672



```python
pca_df_test=pd.DataFrame()    #PCA를 통해 차원 축소한 결과를 
pca_df_test['pca-one'] = pca_test_result[:,0]
pca_df_test['pca-two'] = pca_test_result[:,1] 
pca_df_test['pca-three'] = pca_test_result[:,2] 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for i in range(10):
    ax.scatter(pca_df_test['pca-one'][df_test['y']==i]
               , pca_df_test['pca-two'][df_test['y']==i]
               , pca_df_test['pca-three'][df_test['y']==i]
               , s = 10)
ax.set_xlabel('PCA - ONE')
ax.set_ylabel('PCA - Two')
ax.set_zlabel('PCA - Three')
plt.show()
```

    Explained variation per principal component: [0.09827797 0.07144009 0.06202222]



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_17_1.png)


# LDA


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
time_start = time.time()
lda = LinearDiscriminantAnalysis(n_components=2)
lda_result = lda.fit_transform(df_train[feat_cols].values, df_train['y'])
print( 't-SNE done! Time elapsed: {} seconds',time.time() - time_start )
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_19_1.png)



```python
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('First and Second LDA colored by digit', fontsize = 20)

for i in range(10):
    ax.scatter(lda_result[df_train['y']==i,0]
            ,lda_result[df_train['y']==i,1]
            , s = 10)

ax.legend(range(10))
ax.grid()
```

- label별로 나눠져 있는 것을 알 수 있습니다.

## Fit 시켰던 Model을 이용해서 Test Data Transform하기 & 2차원 평면에 나타내기


```python
time_start = time.time()
lda_result = lda.transform(df_test[feat_cols].values)
print( 't-SNE done! Time elapsed: {} seconds',time.time() - time_start )
```

    t-SNE done! Time elapsed: {} seconds 0.11358332633972168



```python
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xticks((-6,8))
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('First and Second LDA colored by digit', fontsize = 20)
plt.xlim(-6,8)
plt.ylim(-6,8)
for i in range(10):
    ax.scatter(lda_result[df_test['y']==i,0]
            ,lda_result[df_test['y']==i,1]
            , s = 10)

ax.legend(range(10))
ax.grid()
```


![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_24_0.png)


### lda의 accuracy 측정



```python
lda.score(df_test[feat_cols].values,df_test['y'])
```




    0.8465



- label에 따라 잘 구분 된것 같습니다.

# 다항분류기 적용

## Random Forest 적용


```python
from sklearn.ensemble import RandomForestClassifier
eclf = RandomForestClassifier(n_estimators=100,
                              max_features=2,
                              n_jobs=-1, oob_score=True)
```


```python
eclf.fit(df_train[feat_cols],df_train['y'])
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features=2, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=True, random_state=None, verbose=0,
                           warm_start=False)




```python
eclf.score(df_test[feat_cols],df_test['y'])
```




    0.935



- 정확도 측정


```python
eclf.predict(df_test.loc[df_test['y']==1][feat_cols])
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 6., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 2.])



- 정말로 잘 분류하는지 확인해보기. 육안상 오차는 3개정도 입니다.

## AdaBoost 적용


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
eclf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=500, 
                          learning_rate=0.1)
```


```python
eclf.fit(df_train[feat_cols],df_train['y'])
```




    AdaBoostClassifier(algorithm='SAMME.R',
                       base_estimator=DecisionTreeClassifier(class_weight=None,
                                                             criterion='gini',
                                                             max_depth=2,
                                                             max_features=None,
                                                             max_leaf_nodes=None,
                                                             min_impurity_decrease=0.0,
                                                             min_impurity_split=None,
                                                             min_samples_leaf=1,
                                                             min_samples_split=2,
                                                             min_weight_fraction_leaf=0.0,
                                                             presort=False,
                                                             random_state=None,
                                                             splitter='best'),
                       learning_rate=0.1, n_estimators=500, random_state=None)




```python
eclf.score(df_test[feat_cols],df_test['y'])
```




    0.8305




```python
eclf.predict(df_test.loc[df_test['y']==1][feat_cols])
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 7.,
           1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 8., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1.,
           1., 1., 1., 1., 8., 1., 1., 1., 1., 3.])



#### Radom Forest의 성능이 AdaBoost보다 좋게 나온 것에 대한 해석

- Random Forest에서의 각 트리에서는 변수를 임의로 선택합니다.
- 이 데이터의 경우, 설명변수가 700개 이상으로 매우 많습니다.
- 설명변수가 많을 경우, 대체로 변수간 상관성이 높은 변수가 섞일 확률이 높습니다.
- 특히 Mnist 데이터에서는 글씨이기 때문에, 
- 하나의 픽셀은 주변 픽셀과 값이 같을 확률이 크기 때문에,
- 하나의 변수가 주변 변수와 같은 값을 가질 확률이 매우 큽니다.
- 변수를 랜덤하게 뽑는 Random Forest는 변수간 상관성을 줄여줄 수 있습니다.

# TSNE


```python
import time
from sklearn.manifold import TSNE

time_start = time.time()

tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df_train[feat_cols].values)

print( 't-SNE done! Time elapsed: {} seconds',time.time() - time_start )

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('TSNE 1', fontsize = 15)
ax.set_ylabel('TSNE 2', fontsize = 15)
ax.set_title('First and Second TSNE colored by digit', fontsize = 20)

for i in range(10):
    ax.scatter(tsne_results[df_train['y']==i,0]
            ,tsne_results[df_train['y']==i,1]
            , s = 10)

ax.legend(range(10))
ax.grid()
```

    t-SNE done! Time elapsed: {} seconds 300.47662830352783



![png](week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_DR_%EA%B8%B8%ED%83%9C%ED%98%95_45_1.png)

