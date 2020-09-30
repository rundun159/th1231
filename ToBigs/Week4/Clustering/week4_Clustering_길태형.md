```python
from sklearn import datasets
import pandas as pd
import numpy as np
data = pd.read_csv('Mall_Customers.csv')
```


```python
data.isnull().sum()
```




    CustomerID                0
    Gender                    0
    Age                       0
    Annual Income (k$)        0
    Spending Score (1-100)    0
    dtype: int64



- 결측치 확인


```python
data.describe()
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
      <th>CustomerID</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>100.500000</td>
      <td>38.850000</td>
      <td>60.560000</td>
      <td>50.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>57.879185</td>
      <td>13.969007</td>
      <td>26.264721</td>
      <td>25.823522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50.750000</td>
      <td>28.750000</td>
      <td>41.500000</td>
      <td>34.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.500000</td>
      <td>36.000000</td>
      <td>61.500000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>150.250000</td>
      <td>49.000000</td>
      <td>78.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>200.000000</td>
      <td>70.000000</td>
      <td>137.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



- Data 분포 확인


```python
data=data.drop(['CustomerID'],axis=1)
```

- CustomerID는 중요한 feature가 아니므로 삭제하겠습니다.


```python
data['Gender_int']=np.empty(len(data),int)
```


```python
data.loc[data['Gender']=='Male','Gender_int']=1
data.loc[data['Gender']=='Female','Gender_int']=0
```


```python
data=data.drop(['Gender'],axis=1)
```

- Gender를 0,1 encoding으로 변환
- 0: Female, 1: Male


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
data_Scaled = scaler.fit_transform(data)
```


```python
df = pd.DataFrame(data_Scaled)
```


```python
df.columns=['Age','Annual Income (k$)','Spending Score (1-100)','Gender_int']
```


```python
df.describe()
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
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
      <th>Gender_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000e+02</td>
      <td>2.000000e+02</td>
      <td>2.000000e+02</td>
      <td>2.000000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.021405e-16</td>
      <td>-2.131628e-16</td>
      <td>-1.465494e-16</td>
      <td>3.108624e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.002509e+00</td>
      <td>1.002509e+00</td>
      <td>1.002509e+00</td>
      <td>1.002509e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.496335e+00</td>
      <td>-1.738999e+00</td>
      <td>-1.910021e+00</td>
      <td>-8.864053e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.248436e-01</td>
      <td>-7.275093e-01</td>
      <td>-5.997931e-01</td>
      <td>-8.864053e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.045351e-01</td>
      <td>3.587926e-02</td>
      <td>-7.764312e-03</td>
      <td>-8.864053e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.284319e-01</td>
      <td>6.656748e-01</td>
      <td>8.851316e-01</td>
      <td>1.128152e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.235532e+00</td>
      <td>2.917671e+00</td>
      <td>1.894492e+00</td>
      <td>1.128152e+00</td>
    </tr>
  </tbody>
</table>
</div>



## Target Data 따로 저장하기 & Encoding

- Spending Score(1-100) :Score assigned by the mall based on customer behavior and spending nature
- Mall입장에서는 이 Score이 가장 중요할것이므로, 이 Feature를 Target Data로 지정하겠습니다.


```python
data_labels = pd.DataFrame(data['Spending Score (1-100)'])
```


```python
data_labels.describe()
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
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25.823522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



#####  4분위 수를 이용해서 Encoding 하려고 합니다.


```python
data_labels['labels']=np.array(len(data))
data_labels.loc[data_labels['Spending Score (1-100)']<34.750000,'labels']=0
data_labels.loc[(data_labels['Spending Score (1-100)']>=34.750000)&(data_labels['Spending Score (1-100)']<50.000000),'labels']=1
data_labels.loc[(data_labels['Spending Score (1-100)']>=50.000000)&(data_labels['Spending Score (1-100)']<73.000000),'labels']=2
data_labels.loc[(data_labels['Spending Score (1-100)']>=73.000000),'labels']=3
```


```python
data_labels=data_labels.drop(['Spending Score (1-100)'],axis=1)
```


```python
data_labels.head()
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
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



##  Data set의 3가지 종류
### data : Scaling 안된 Data set


```python
data.pop('Spending Score (1-100)')
```




    0      39
    1      81
    2       6
    3      77
    4      40
           ..
    195    79
    196    28
    197    74
    198    18
    199    83
    Name: Spending Score (1-100), Length: 200, dtype: int64




```python
df.pop('Spending Score (1-100)')
```




    0     -0.434801
    1      1.195704
    2     -1.715913
    3      1.040418
    4     -0.395980
             ...   
    195    1.118061
    196   -0.861839
    197    0.923953
    198   -1.250054
    199    1.273347
    Name: Spending Score (1-100), Length: 200, dtype: float64




```python
data.head() 
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
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Gender_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>17</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### df: Scaling 된 Data set


```python
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
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Gender_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.424569</td>
      <td>-1.738999</td>
      <td>1.128152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.281035</td>
      <td>-1.738999</td>
      <td>1.128152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.352802</td>
      <td>-1.700830</td>
      <td>-0.886405</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.137502</td>
      <td>-1.700830</td>
      <td>-0.886405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.563369</td>
      <td>-1.662660</td>
      <td>-0.886405</td>
    </tr>
  </tbody>
</table>
</div>



### data_labels : Encoding을 거친 Target Data


```python
data_labels.head()
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
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 1. Hierarchical Clustering


```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
```

## Scaling 안 된 데이터


```python
# Average으로 군집-군집 or 군집-개체 간 거리 계산 
links = linkage(data,method='average')
# Plot the dendrogram
plt.figure(figsize=(40,20))
dendrogram(links,
           labels = data_labels.as_matrix(columns=['labels']),
           leaf_rotation=90,
           leaf_font_size=20,
)
plt.show()
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\matplotlib\text.py:1150: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if s != self._text:



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_34_1.png)


## Scaling 된 데이터


```python
# MAX(Complete Link)으로 군집-군집 or 군집-개체 간 거리 계산 
links = linkage(df,method='average')
# Plot the dendrogram
plt.figure(figsize=(40,20))
dendrogram(links,
           labels = data_labels.as_matrix(columns=['labels']),
           leaf_rotation=90,
           leaf_font_size=20,
)
plt.show()
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\matplotlib\text.py:1150: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if s != self._text:



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_36_1.png)


# Evaluation1_Gini Index

- 클러스터링을 마쳤을때, 각 군집이 label별로 모여있는 게
- 이 문맥에서 좋은 클러스터링이라고 생각했습니다.
- 그것을 표현할 수 있는 지표가 Gini 계수라고 생각해서
- Decision Tree는 아니지만, 도입하게 되었습니다.


```python
def getGini(npArr):
    gini=1
    sizeofArr = npArr.sum()
    for i in range(0,len(npArr)):
        gini-=(npArr[i]/sizeofArr)*(npArr[i]/sizeofArr)
    return gini
```


```python
def getTotalGini(df,totalSize):
    #crosstab이 df로 주어짐.
    gini=0
    for i in range(0,len(df)):
        gini+=(df.iloc[i].sum()/totalSize)*getGini(np.array(df.iloc[i]))
    return gini
```


```python
def retBestCluster_linkage(df,labels):
    methods=['average','centroid','complete','median','single','ward','weighted']
    ret=pd.DataFrame()
    minGini = 1
    bestMethod =''
    bestT=0
    for met in methods:
        links = linkage(df,method=met)
        for t in range(1,100):
            predict = pd.DataFrame(fcluster(links,t,criterion='distance'))
            predict.columns=['predict']
            ct = pd.crosstab(predict['predict'],labels['labels'])
            if (len(ct)>(len(labels['labels'].unique())-3)) and (len(ct)<(len(labels['labels'].unique())+3)):
                    ctGini = getTotalGini(ct,len(df))
                    if ctGini<minGini:
                        minGini=ctGini
                        ret=ct
                        bestMethod=met
                        bestT=t
    return ret, minGini, bestMethod, bestT
```

## Scaling 안 된 데이터


```python
best=retBestCluster_linkage(data,data_labels)
```


```python
best
```




    (labels    0   1   2   3
     predict                
     1         3   8  10  18
     2        14   5   6   0
     3         0  22  18   0
     4         8   1   2   9
     5         7   1   0   0
     6        18  11  15  24, 0.6235144042232277, 'average', 26)




```python
#Gini 값이 최소일때의 t값
best[3]
```




    26




```python
getTotalGini(best[0],len(data))
```




    0.6235144042232277



## Scaling 된 데이터


```python
best=retBestCluster_linkage(df,data_labels)
```


```python
best
```




    (labels    0   1   2   3
     predict                
     1         6  12   8   0
     2        17   2   3  15
     3         1   6  11   7
     4         4  11  13  18
     5         9  12  13   0
     6        13   5   3  11, 0.6634162695205534, 'weighted', 2)




```python
#Gini 값이 최소일때의 t값
best[3]
```




    2




```python
getTotalGini(best[0],len(df))
```




    0.6634162695205534



- t값, threshold값을 비교했을떄에는, scaling을 거친 df 데이터의 t값이 더 작은것을 볼 수 있습니다.

- crosstab을 봤을때, 하나의 클러스터의 데이터들이 하나의 label만 갖는것이 아니라
- 여러 label 값을 균등하게 갖는것을 볼 수 있습니다.
- 한 label에 모여있는 형태도 gini index값이 작겠지만,
- 균등하게 퍼져있는 형태도 gini index의 값이 작게 나오기 때문에, 이런 결과값이 도출된 것 같습니다.
- 또한, 클러스터의 갯수를 선정하는것도 쉽지 않았습니다.
- Gini Index로 측정한 한계라고 생각합니다.

- 주어진 Feature와 target Data의 상관관계가 정확하게 밝혀지지 않은 상태에서는
- Gini Index로 클러스터링의 성능을 판단하는 것은 무리가 있다고 생각합니다.

# Evaluation2_silhouette


```python
from sklearn.metrics import silhouette_score, silhouette_samples
```


```python
def plotSil_fcluster(data):
    newData=data.copy()
    sil = {}
    for t in range(2, 100):
        links = linkage(newData,method='centroid')
        predict = pd.DataFrame(fcluster(links,t,criterion='distance'))
        if(len(predict[0].unique())==1):
            continue
        sil[t] = silhouette_score(newData, predict, metric='euclidean')
    plt.figure()
    plt.plot(list(sil.keys()), list(sil.values()))
    plt.xlabel("valueOfThreshold")
    plt.ylabel("Silhouette_Fcluster")
    plt.show()
    del newData
```

## Scaling 안 된 데이터


```python
plotSil_fcluster(data)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_59_1.png)



```python
plotSil_fcluster(df)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_60_1.png)


- Gini Index를 통해서 구한 최적의 threshold 값과, 
- Silhouette를 통해서 구한 최적의 threshold 값이 유사합니다.
- 결과만 놓고 봤을 때에는, Gini Index를 통해 성능을 측정하는 것도 괜찮은 방법이었다고 생각됩니다.

- GIni Index를 통해서도, sihouette을 통해서도,
- scaling을 거치지 않은 데이터의 Evalutaion이 좋게 나왔습니다.
- 이 데이터의 특성상,
- income의 영향력이 결과에 영향이 클거라고 예상되는데,
- scailng을 거치지 않은 data의 클러스터링 결과가 좋은 것 같습니다.

# 2. K-Means Clustering


```python
from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt
import seaborn as sns
```

## Scaling 안된 데이터 클러스터링


```python
#K-Means 군집 분석 
model = KMeans(n_clusters=4,algorithm='auto')
model.fit(data)
data_predict = pd.DataFrame(model.predict(data))
data_predict.columns=['predict']

# predict 추가 
r = pd.concat([data,data_predict],axis=1)

print(r)
# 각 군집의 중심점 
centers = pd.DataFrame(model.cluster_centers_,columns=['Age','Annual Income (k$)','Gender_int'])
center_x = centers['Age']
center_y = centers['Annual Income (k$)']
center_z = centers['Gender_int']

from mpl_toolkits.mplot3d import Axes3D
# scatter plot
fig = plt.figure( figsize=(9,9))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(r['Age'],r['Annual Income (k$)'],r['Gender_int'],c=r['predict'],alpha=0.5)
ax.scatter(center_x,center_y,center_z,s=50,marker='D',c='r')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Gender_int')
plt.show()

```

         Age  Annual Income (k$)  Gender_int  predict
    0     19                  15           1        2
    1     21                  15           1        2
    2     20                  16           0        2
    3     23                  16           0        2
    4     31                  17           0        2
    ..   ...                 ...         ...      ...
    195   35                 120           0        3
    196   45                 126           0        3
    197   32                 126           1        3
    198   32                 137           1        3
    199   30                 137           1        3
    
    [200 rows x 4 columns]



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_66_1.png)


## Scaling 된 데이터 클러스터링


```python
model = KMeans(n_clusters=4,algorithm='auto')
model.fit(df)
df_predict = pd.DataFrame(model.predict(df))
df_predict.columns=['predict']

# predict 추가 
r = pd.concat([df,df_predict],axis=1)

print(r)
# 각 군집의 중심점 
centers = pd.DataFrame(model.cluster_centers_,columns=['Age','Annual Income (k$)','Gender_int'])
center_x = centers['Age']
center_y = centers['Annual Income (k$)']
center_z = centers['Gender_int']
from mpl_toolkits.mplot3d import Axes3D
# scatter plot
fig = plt.figure( figsize=(9,9))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(r['Age'],r['Annual Income (k$)'],r['Gender_int'],c=r['predict'],alpha=0.5)
ax.scatter(center_x,center_y,center_z,s=50,marker='D',c='r')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Gender_int')
plt.show()

```

              Age  Annual Income (k$)  Gender_int  predict
    0   -1.424569           -1.738999    1.128152        3
    1   -1.281035           -1.738999    1.128152        3
    2   -1.352802           -1.700830   -0.886405        0
    3   -1.137502           -1.700830   -0.886405        0
    4   -0.563369           -1.662660   -0.886405        0
    ..        ...                 ...         ...      ...
    195 -0.276302            2.268791   -0.886405        2
    196  0.441365            2.497807   -0.886405        2
    197 -0.491602            2.497807    1.128152        3
    198 -0.491602            2.917671    1.128152        3
    199 -0.635135            2.917671    1.128152        3
    
    [200 rows x 4 columns]



![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_68_1.png)


- 육안 상으로 봤을 때에는... 스케일링 된 데이터 셋이 더 잘 클러스터링 된것 같습니다.

# Evaluation 1_ Gini index 


```python
def getTotalGini(df,totalSize):
    #crosstab이 df로 주어짐.
    gini=0
    for i in range(0,len(df)):
        gini+=(df.iloc[i].sum()/totalSize)*getGini(np.array(df.iloc[i]))
    return gini
def getGini(npArr):
    gini=1
    sizeofArr = npArr.sum()
    for i in range(0,len(npArr)):
        gini-=(npArr[i]/sizeofArr)*(npArr[i]/sizeofArr)
    return gini
```

## Scaling 안 된 데이터


```python
ct = pd.crosstab(data_predict['predict'],data_labels['labels'])
getTotalGini(ct,len(data_predict))
```




    0.6830914717485166




```python
ct.head(100)
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
      <th>labels</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>predict</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>25</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>14</td>
      <td>16</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>8</td>
      <td>11</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



## Scaling 된 데이터


```python
ct = pd.crosstab(df_predict['predict'],data_labels['labels'])
getTotalGini(ct,len(df_predict))
```




    0.7142078819710398




```python
ct.head(100)
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
      <th>labels</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>predict</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>15</td>
      <td>17</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>16</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>10</td>
      <td>9</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>7</td>
      <td>14</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>



- Gini index로 판단했을때, scaling을 거치지 않은 data set의 값이 더 작으므로,
- Scaling 안된 데이터로 클러스터링 하는게 더 좋은 성능을 발휘한다고 볼 수 있습니다.

# Evaluation2_  SSE


```python
def plotSSE(data):
    newData=data.copy()
    sse = {}
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(newData)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
    del newData
```

## Scaling 안 된 데이터


```python
plotSSE(data)
```


![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_82_0.png)


## Scaling 된 데이터


```python
plotSSE(df)
```


![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_84_0.png)


- 차이 값이 절대적으로 큰 scaling이 안된 data의 SSE의 값이 큰것을 볼 수 있습니다.
- 따라서 scaling 여부에 따른 성능 개선 여부는 판단하기 어렵습니다.

- scaling된 데이터의 elbow는 6이고, scaling 되지 않은 데이터의 elbow는 4라고 볼 수 있습니다.

# Evaluation3_silhouette


```python
from sklearn.metrics import silhouette_score, silhouette_samples
```


```python
def plotSil(data):
    newData=data.copy()
    sil = {}
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(newData)
        sil[k] = silhouette_score(newData, kmeans.labels_, metric='euclidean')
    plt.figure()
    plt.plot(list(sil.keys()), list(sil.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette")
    plt.show()
    del newData
```

## Scaling 안 된 데이터


```python
plotSil(data)
```


![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_91_0.png)


## Scaling 된 데이터


```python
plotSil(df)
```


![png](week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_files/week4_Clustering_%EA%B8%B8%ED%83%9C%ED%98%95_93_0.png)


- silhouette으로 성능을 판단했을때 

- scaling을 거치지 않은 data 셋의 실루엣 값의 최대는 0.43보다 크고,
- scaling을 거친 data셋의 실루엣 값의 최대는 0.43보다 작습니다.
- 게다가, scacling을 거친 data set은 클러스터의 갯수가 6이상이어야 좋은 성능을 발휘합니다.
- 클러스터의 갯수가 6보다 커지게 되면 label의 class 갯수보다 커지게 되므로 
- 유의미하지 않다고 생각합니다.

- 반면, scaling을 거치지 않은 data set은 클러스터의 갯수가 4일때 좋은 성능을 발휘합니다.
- label의 class 갯수와 동일하므로 좋은 클러스터링이라고 생각합니다.

# 종합

## Hierarchical Clustering

## KMeans Clustering

- 두가지 클러스터링 방법을 사용했을때,
- 모든 Evaluation에서 Scaling을 거치지 않은 데이터 셋에서
- 클러스터링이 잘 된 것으로 보입니다.

- 주어진 데이터 셋에서는 Income이 Importance이 큰 변수로 예상됩니다.
- 따라서 Normalize되지 않고 Income의 중요도가 살아있는,
- Scaling 되지 않은 Data set이 클러스터링 잘 되는 것 같습니다.
