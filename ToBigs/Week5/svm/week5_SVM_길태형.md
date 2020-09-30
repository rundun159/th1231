# Santander Customer Transaction Prediction


```python
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
```


```python
df_train = pd.read_csv('./train.csv')
df_train.head()
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
      <th>ID_code</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>...</th>
      <th>var_190</th>
      <th>var_191</th>
      <th>var_192</th>
      <th>var_193</th>
      <th>var_194</th>
      <th>var_195</th>
      <th>var_196</th>
      <th>var_197</th>
      <th>var_198</th>
      <th>var_199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train_0</td>
      <td>0</td>
      <td>8.9255</td>
      <td>-6.7863</td>
      <td>11.9081</td>
      <td>5.0930</td>
      <td>11.4607</td>
      <td>-9.2834</td>
      <td>5.1187</td>
      <td>18.6266</td>
      <td>...</td>
      <td>4.4354</td>
      <td>3.9642</td>
      <td>3.1364</td>
      <td>1.6910</td>
      <td>18.5227</td>
      <td>-2.3978</td>
      <td>7.8784</td>
      <td>8.5635</td>
      <td>12.7803</td>
      <td>-1.0914</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train_1</td>
      <td>0</td>
      <td>11.5006</td>
      <td>-4.1473</td>
      <td>13.8588</td>
      <td>5.3890</td>
      <td>12.3622</td>
      <td>7.0433</td>
      <td>5.6208</td>
      <td>16.5338</td>
      <td>...</td>
      <td>7.6421</td>
      <td>7.7214</td>
      <td>2.5837</td>
      <td>10.9516</td>
      <td>15.4305</td>
      <td>2.0339</td>
      <td>8.1267</td>
      <td>8.7889</td>
      <td>18.3560</td>
      <td>1.9518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train_2</td>
      <td>0</td>
      <td>8.6093</td>
      <td>-2.7457</td>
      <td>12.0805</td>
      <td>7.8928</td>
      <td>10.5825</td>
      <td>-9.0837</td>
      <td>6.9427</td>
      <td>14.6155</td>
      <td>...</td>
      <td>2.9057</td>
      <td>9.7905</td>
      <td>1.6704</td>
      <td>1.6858</td>
      <td>21.6042</td>
      <td>3.1417</td>
      <td>-6.5213</td>
      <td>8.2675</td>
      <td>14.7222</td>
      <td>0.3965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train_3</td>
      <td>0</td>
      <td>11.0604</td>
      <td>-2.1518</td>
      <td>8.9522</td>
      <td>7.1957</td>
      <td>12.5846</td>
      <td>-1.8361</td>
      <td>5.8428</td>
      <td>14.9250</td>
      <td>...</td>
      <td>4.4666</td>
      <td>4.7433</td>
      <td>0.7178</td>
      <td>1.4214</td>
      <td>23.0347</td>
      <td>-1.2706</td>
      <td>-2.9275</td>
      <td>10.2922</td>
      <td>17.9697</td>
      <td>-8.9996</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train_4</td>
      <td>0</td>
      <td>9.8369</td>
      <td>-1.4834</td>
      <td>12.8746</td>
      <td>6.6375</td>
      <td>12.2772</td>
      <td>2.4486</td>
      <td>5.9405</td>
      <td>19.2514</td>
      <td>...</td>
      <td>-1.4905</td>
      <td>9.5214</td>
      <td>-0.1508</td>
      <td>9.1942</td>
      <td>13.2876</td>
      <td>-1.5121</td>
      <td>3.9267</td>
      <td>9.5031</td>
      <td>17.9974</td>
      <td>-8.8104</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 202 columns</p>
</div>



### null값과 shape 확인


```python
df_train.isnull().sum().sum()
```




    0




```python
df_train.shape
```




    (200000, 202)



### data 준비


```python
df_train.head()
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
      <th>ID_code</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>...</th>
      <th>var_190</th>
      <th>var_191</th>
      <th>var_192</th>
      <th>var_193</th>
      <th>var_194</th>
      <th>var_195</th>
      <th>var_196</th>
      <th>var_197</th>
      <th>var_198</th>
      <th>var_199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train_0</td>
      <td>0</td>
      <td>8.9255</td>
      <td>-6.7863</td>
      <td>11.9081</td>
      <td>5.0930</td>
      <td>11.4607</td>
      <td>-9.2834</td>
      <td>5.1187</td>
      <td>18.6266</td>
      <td>...</td>
      <td>4.4354</td>
      <td>3.9642</td>
      <td>3.1364</td>
      <td>1.6910</td>
      <td>18.5227</td>
      <td>-2.3978</td>
      <td>7.8784</td>
      <td>8.5635</td>
      <td>12.7803</td>
      <td>-1.0914</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train_1</td>
      <td>0</td>
      <td>11.5006</td>
      <td>-4.1473</td>
      <td>13.8588</td>
      <td>5.3890</td>
      <td>12.3622</td>
      <td>7.0433</td>
      <td>5.6208</td>
      <td>16.5338</td>
      <td>...</td>
      <td>7.6421</td>
      <td>7.7214</td>
      <td>2.5837</td>
      <td>10.9516</td>
      <td>15.4305</td>
      <td>2.0339</td>
      <td>8.1267</td>
      <td>8.7889</td>
      <td>18.3560</td>
      <td>1.9518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train_2</td>
      <td>0</td>
      <td>8.6093</td>
      <td>-2.7457</td>
      <td>12.0805</td>
      <td>7.8928</td>
      <td>10.5825</td>
      <td>-9.0837</td>
      <td>6.9427</td>
      <td>14.6155</td>
      <td>...</td>
      <td>2.9057</td>
      <td>9.7905</td>
      <td>1.6704</td>
      <td>1.6858</td>
      <td>21.6042</td>
      <td>3.1417</td>
      <td>-6.5213</td>
      <td>8.2675</td>
      <td>14.7222</td>
      <td>0.3965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train_3</td>
      <td>0</td>
      <td>11.0604</td>
      <td>-2.1518</td>
      <td>8.9522</td>
      <td>7.1957</td>
      <td>12.5846</td>
      <td>-1.8361</td>
      <td>5.8428</td>
      <td>14.9250</td>
      <td>...</td>
      <td>4.4666</td>
      <td>4.7433</td>
      <td>0.7178</td>
      <td>1.4214</td>
      <td>23.0347</td>
      <td>-1.2706</td>
      <td>-2.9275</td>
      <td>10.2922</td>
      <td>17.9697</td>
      <td>-8.9996</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train_4</td>
      <td>0</td>
      <td>9.8369</td>
      <td>-1.4834</td>
      <td>12.8746</td>
      <td>6.6375</td>
      <td>12.2772</td>
      <td>2.4486</td>
      <td>5.9405</td>
      <td>19.2514</td>
      <td>...</td>
      <td>-1.4905</td>
      <td>9.5214</td>
      <td>-0.1508</td>
      <td>9.1942</td>
      <td>13.2876</td>
      <td>-1.5121</td>
      <td>3.9267</td>
      <td>9.5031</td>
      <td>17.9974</td>
      <td>-8.8104</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 202 columns</p>
</div>




```python
y=df_train['target']
```


```python
X=df_train.drop(['target','ID_code'],axis=1)
```

- 랜덤하게 뽑기


```python
a=np.random.choice(70000, 10000,replace=False)
```


```python
X=X.iloc[a]
y=y.iloc[a]
```


```python
#데이터 scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```


```python
X_scaled.shape
```




    (10000, 200)



- X_scaled에 Scaling 된 데이터 저장.

### SVM으로 학습


```python
from sklearn.svm import SVC
from sklearn import metrics #model evaluation 
```


```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```


```python
for i in range(1,10):
    pca = PCA(n_components=i)
    pca_features=pca.fit_transform(X)
    #train / test split
    X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=1)
    #default linear
    svc=SVC(kernel='linear')
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score when'+ '# of Comp is '+str(i)+': ' )
    print(metrics.accuracy_score(y_test,y_pred))
```

- Scaling을 거치지 않은 데이터 활용


```python
for i in range(1,10):
    pca = PCA(n_components=i)
    pca_features=pca.fit_transform(X_scaled)
    #train / test split
    X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=1)
    #default linear
    svc=SVC(kernel='linear')
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score when'+ '# of Comp is '+str(i)+': ' )
    print(metrics.accuracy_score(y_test,y_pred))
```

- PCA의 차원수에 따른 정확도 비교
- PCA의 차원수와 관계 없이 일정한 결과를 보여줍니다.

- Data의 Scaling여부와 SVC정확도에는 관련이 없어보입니다.

# SVC 모델 종류에 따른 정확도 비교 (K-Fold Validation 사용)


```python
pca = PCA(n_components=3)
pca_features=pca.fit_transform(X_scaled)
#train / test split
X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=1)
```


```python
#default linear
from sklearn.model_selection import cross_validate

svc=SVC(kernel='linear')
scores = cross_validate(svc, X_train, y_train, cv=4, scoring='accuracy') #cv is cross validation
np.mean(scores['test_score']) #교차검증 성능 평균(test set)
```




    0.9001250500312625




```python
#default RBF kernel
from sklearn.model_selection import cross_validate

svc=SVC(kernel='rbf')
scores = cross_validate(svc, X_train, y_train, cv=4, scoring='accuracy') #cv is cross validation
np.mean(scores['test_score']) #교차검증 성능 평균(test set)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    0.9001250500312625




```python
#default polynomial kernel
from sklearn.model_selection import cross_validate

svc=SVC(kernel='poly')
scores = cross_validate(svc, X_train, y_train, cv=4, scoring='accuracy') #cv is cross validation
np.mean(scores['test_score']) #교차검증 성능 평균(test set)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    0.9001250500312625



- 왜 스코어가 같게 나오는지 잘 모르겠습니다...

# Hyperparameter search

### Linear


```python
pca = PCA(n_components=2)
pca_features=pca.fit_transform(X_scaled)
#train / test split
X_train, X_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=1)
```


```python
#CV으로 하이퍼파라미터 알아보기
C_range=list(range(1,26))
acc_score = []
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_validate(svc, X_train, y_train, cv=5, scoring='accuracy')
    acc_score.append(np.mean(scores['test_score']))
```


```python
print(np.mean(acc_score))
```

    0.9001250625488527



```python
import matplotlib.pyplot as plt
%matplotlib inline


C_values=list(range(1,26))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')
```




    Text(0, 0.5, 'Cross-Validated Accuracy')




![png](week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_35_1.png)



```python
len(set(acc_score))
```




    1



- C값에 관계 없이 이렇게 나오는 이유를 아직 잘 모르겠습니다...
- Data set의 크기가 작기 때문이라고 생각합니다.

###  RBF kernel


```python
#RBF kernel의 하이퍼파라미터 C, gamma중 gamma만 해보자
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_validate(svc, pca_features, y, cv=10, scoring='accuracy')
    acc_score.append(np.mean(scores['test_score']))
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.xticks(np.arange(0.0001,100,5))
plt.ylabel('Cross-Validated Accuracy')
```




    Text(0, 0.5, 'Cross-Validated Accuracy')




![png](week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_40_1.png)


- 0~5사이에 최대가 있을것 같습니다>


```python
gamma_range=np.linspace(0.1,5,num=5)
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_validate(svc, pca_features, y, cv=10, scoring='accuracy')
    acc_score.append(np.mean(scores['test_score']))
```


```python

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.xticks(np.linspace(0.1,5,num=5))
plt.ylabel('Cross-Validated Accuracy')
```




    Text(0, 0.5, 'Cross-Validated Accuracy')




![png](week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_files/week5_SVM_%EA%B8%B8%ED%83%9C%ED%98%95_43_1.png)



```python
from sklearn.svm import SVC
svm_model= SVC()
tuned_parameters = {
 'C': (np.arange(0.1,1,0.1)) , 'kernel': ['linear'],
 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf']
                   }
from sklearn.model_selection import GridSearchCV

model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')
model_svm.fit(X_train, y_train)
print(model_svm.best_score_)
```


```python
print(model_svm.best_params_)
```


```python
from sklearn.svm import SVC
svc= SVC(C=0.9, gamma=0.05, kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
```


```python

```
