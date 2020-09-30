```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(1)
```

# diabetes 데이터

# a) 데이터 로드 및 처리


```python
#데이터 로드
from sklearn import datasets
diabetes = datasets.load_diabetes()
```


```python
"""
    sklearn에 있는 당뇨병 진행도 데이터를 사용

    <변수>
    Age
    Sex
    Body mass index
    Average blood pressure
    S1 : 혈청에 대한 6가지 지표들
    S2
    S3
    S4
    S5
    S6
    
    *데이터가 각 컬럼의 합이 1이 되도록 centering, scaling됨

    <데이터>
    diabetes에 data(설명변수), target(종속변수) 데이터가 따로 있음
"""
```




    '\n    sklearn에 있는 당뇨병 진행도 데이터를 사용\n\n    <변수>\n    Age\n    Sex\n    Body mass index\n    Average blood pressure\n    S1 : 혈청에 대한 6가지 지표들\n    S2\n    S3\n    S4\n    S5\n    S6\n    \n    *데이터가 각 컬럼의 합이 1이 되도록 centering, scaling됨\n\n    <데이터>\n    diabetes에 data(설명변수), target(종속변수) 데이터가 따로 있음\n'




```python
#설명변수
diabetes.data
```




    array([[ 0.03807591,  0.05068012,  0.06169621, ..., -0.00259226,
             0.01990842, -0.01764613],
           [-0.00188202, -0.04464164, -0.05147406, ..., -0.03949338,
            -0.06832974, -0.09220405],
           [ 0.08529891,  0.05068012,  0.04445121, ..., -0.00259226,
             0.00286377, -0.02593034],
           ...,
           [ 0.04170844,  0.05068012, -0.01590626, ..., -0.01107952,
            -0.04687948,  0.01549073],
           [-0.04547248, -0.04464164,  0.03906215, ...,  0.02655962,
             0.04452837, -0.02593034],
           [-0.04547248, -0.04464164, -0.0730303 , ..., -0.03949338,
            -0.00421986,  0.00306441]])




```python
diabetes.feature_names
```




    ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']




```python
#array형으로 되어있으므로 다루기 쉽게 데이터프레임으로 바꿔준다.
df_x = pd.DataFrame(diabetes.data)
df_x.columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
df_x.head()
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
  </tbody>
</table>
</div>




```python
#종속변수
df_y = pd.DataFrame(diabetes.target, columns=['target'])
df_y.head()
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
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.concat([df_x,df_y],axis=1)
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>206.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>135.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#442개의 관측치와 10개의 설명변수, 1개의 타겟변수로 이루어져있다.
df.shape
```




    (442, 11)



# b) EDA


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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>442.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.639623e-16</td>
      <td>1.309912e-16</td>
      <td>-8.013951e-16</td>
      <td>1.289818e-16</td>
      <td>-9.042540e-17</td>
      <td>1.301121e-16</td>
      <td>-4.563971e-16</td>
      <td>3.863174e-16</td>
      <td>-3.848103e-16</td>
      <td>-3.398488e-16</td>
      <td>152.133484</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>77.093005</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.072256e-01</td>
      <td>-4.464164e-02</td>
      <td>-9.027530e-02</td>
      <td>-1.123996e-01</td>
      <td>-1.267807e-01</td>
      <td>-1.156131e-01</td>
      <td>-1.023071e-01</td>
      <td>-7.639450e-02</td>
      <td>-1.260974e-01</td>
      <td>-1.377672e-01</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-3.729927e-02</td>
      <td>-4.464164e-02</td>
      <td>-3.422907e-02</td>
      <td>-3.665645e-02</td>
      <td>-3.424784e-02</td>
      <td>-3.035840e-02</td>
      <td>-3.511716e-02</td>
      <td>-3.949338e-02</td>
      <td>-3.324879e-02</td>
      <td>-3.317903e-02</td>
      <td>87.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.383060e-03</td>
      <td>-4.464164e-02</td>
      <td>-7.283766e-03</td>
      <td>-5.670611e-03</td>
      <td>-4.320866e-03</td>
      <td>-3.819065e-03</td>
      <td>-6.584468e-03</td>
      <td>-2.592262e-03</td>
      <td>-1.947634e-03</td>
      <td>-1.077698e-03</td>
      <td>140.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.807591e-02</td>
      <td>5.068012e-02</td>
      <td>3.124802e-02</td>
      <td>3.564384e-02</td>
      <td>2.835801e-02</td>
      <td>2.984439e-02</td>
      <td>2.931150e-02</td>
      <td>3.430886e-02</td>
      <td>3.243323e-02</td>
      <td>2.791705e-02</td>
      <td>211.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.107267e-01</td>
      <td>5.068012e-02</td>
      <td>1.705552e-01</td>
      <td>1.320442e-01</td>
      <td>1.539137e-01</td>
      <td>1.987880e-01</td>
      <td>1.811791e-01</td>
      <td>1.852344e-01</td>
      <td>1.335990e-01</td>
      <td>1.356118e-01</td>
      <td>346.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
    age: 이산형
    sex: 범주형(명목형)
    bmi: 연속형
    bp:  연속형
    s1~6:연속형
"""
```




    '\n    age: 이산형\n    sex: 범주형(명목형)\n    bmi: 연속형\n    bp:  연속형\n    s1~6:연속형\n'



# <과제>

이산형 데이터의 unique한 값이 58개 밖에 없는데, 이를 적절하게 인코딩해보세요!

이 변수 그대로 사용해도 되고, 또는 (정확한 나이는 모르지만 ) 나이별 인코딩을 할 수 있을거에요 


```python
"""순서형 인코딩 과제"""

```




    '순서형 인코딩 과제'



- 'age' 


```python
np.percentile(df['age'], 25)
```




    -0.0372992664252317




```python
np.percentile(df['age'], 50)
```




    0.00538306037424807




```python
np.percentile(df['age'], 75)
```




    0.0380759064334241



- 'age_class'라는 feature를 생성해서 4분위 마다 0,1,2,3 값을 넣어주려고 합니다.


```python
df['age_class']=0
```


```python
df.loc[df['age']<-0.0372992664252317,'age_class']=0
```


```python
df.loc[(df['age']<0.00538306037424807)&(df['age']>=-0.0372992664252317),'age_class']=1
```


```python
df.loc[(df['age']>=0.00538306037424807)&(df['age']<0.0380759064334241),'age_class']=2
```


```python
df.loc[df['age']>=0.0380759064334241,'age_class']=3
```


```python
df['age_class'].unique()
```




    array([3, 1, 0, 2], dtype=int64)




```python
df['sex'].head(24)
```




    0     0.050680
    1    -0.044642
    2     0.050680
    3    -0.044642
    4    -0.044642
    5    -0.044642
    6     0.050680
    7     0.050680
    8     0.050680
    9    -0.044642
    10   -0.044642
    11    0.050680
    12   -0.044642
    13    0.050680
    14   -0.044642
    15    0.050680
    16   -0.044642
    17    0.050680
    18   -0.044642
    19   -0.044642
    20   -0.044642
    21    0.050680
    22   -0.044642
    23    0.050680
    Name: sex, dtype: float64




```python
#명목형 인코딩
s_dummy = pd.get_dummies(df.sex, columns=['sex0','sex1'])
s_dummy.columns=['sex0','sex1']
s_dummy.head()
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
      <th>sex0</th>
      <th>sex1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['sex'], axis=1, inplace=True)
```


```python
df = pd.concat([df, s_dummy],axis=1)
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
      <th>sex0</th>
      <th>sex1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>206.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>135.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_x = df.drop(['target'],axis=1)
df_y = pd.DataFrame(df['target'],columns=['target'])
```


```python
print(df_x.columns.values)
print(df_y.columns.values)
```

    ['age' 'bmi' 'bp' 's1' 's2' 's3' 's4' 's5' 's6' 'sex0' 'sex1']
    ['target']



```python
#산점도 행렬
sns.pairplot(df_x)

# s1과 s2의 선형관계가 두드러짐!(다른 s변수들도 살짝 보임)
# s4변수가 이상하게 줄무늬가 보인다
# s4변수의 특성으로 특정 구간의 시작(끝)에 많을 수도 있고, 데이터 기입의 오류일 수도 있겠다
```




    <seaborn.axisgrid.PairGrid at 0x2f5222004a8>




![png](week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_33_1.png)



```python
#상관계수 행렬
df_x.corr()
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>sex0</th>
      <th>sex1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.185085</td>
      <td>0.335427</td>
      <td>0.260061</td>
      <td>0.219243</td>
      <td>-0.075181</td>
      <td>0.203841</td>
      <td>0.270777</td>
      <td>0.301731</td>
      <td>-0.173737</td>
      <td>0.173737</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.185085</td>
      <td>1.000000</td>
      <td>0.395415</td>
      <td>0.249777</td>
      <td>0.261170</td>
      <td>-0.366811</td>
      <td>0.413807</td>
      <td>0.446159</td>
      <td>0.388680</td>
      <td>-0.088161</td>
      <td>0.088161</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>0.335427</td>
      <td>0.395415</td>
      <td>1.000000</td>
      <td>0.242470</td>
      <td>0.185558</td>
      <td>-0.178761</td>
      <td>0.257653</td>
      <td>0.393478</td>
      <td>0.390429</td>
      <td>-0.241013</td>
      <td>0.241013</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>0.260061</td>
      <td>0.249777</td>
      <td>0.242470</td>
      <td>1.000000</td>
      <td>0.896663</td>
      <td>0.051519</td>
      <td>0.542207</td>
      <td>0.515501</td>
      <td>0.325717</td>
      <td>-0.035277</td>
      <td>0.035277</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>0.219243</td>
      <td>0.261170</td>
      <td>0.185558</td>
      <td>0.896663</td>
      <td>1.000000</td>
      <td>-0.196455</td>
      <td>0.659817</td>
      <td>0.318353</td>
      <td>0.290600</td>
      <td>-0.142637</td>
      <td>0.142637</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>-0.075181</td>
      <td>-0.366811</td>
      <td>-0.178761</td>
      <td>0.051519</td>
      <td>-0.196455</td>
      <td>1.000000</td>
      <td>-0.738493</td>
      <td>-0.398577</td>
      <td>-0.273697</td>
      <td>0.379090</td>
      <td>-0.379090</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>0.203841</td>
      <td>0.413807</td>
      <td>0.257653</td>
      <td>0.542207</td>
      <td>0.659817</td>
      <td>-0.738493</td>
      <td>1.000000</td>
      <td>0.617857</td>
      <td>0.417212</td>
      <td>-0.332115</td>
      <td>0.332115</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>0.270777</td>
      <td>0.446159</td>
      <td>0.393478</td>
      <td>0.515501</td>
      <td>0.318353</td>
      <td>-0.398577</td>
      <td>0.617857</td>
      <td>1.000000</td>
      <td>0.464670</td>
      <td>-0.149918</td>
      <td>0.149918</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>0.301731</td>
      <td>0.388680</td>
      <td>0.390429</td>
      <td>0.325717</td>
      <td>0.290600</td>
      <td>-0.273697</td>
      <td>0.417212</td>
      <td>0.464670</td>
      <td>1.000000</td>
      <td>-0.208133</td>
      <td>0.208133</td>
    </tr>
    <tr>
      <th>sex0</th>
      <td>-0.173737</td>
      <td>-0.088161</td>
      <td>-0.241013</td>
      <td>-0.035277</td>
      <td>-0.142637</td>
      <td>0.379090</td>
      <td>-0.332115</td>
      <td>-0.149918</td>
      <td>-0.208133</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>sex1</th>
      <td>0.173737</td>
      <td>0.088161</td>
      <td>0.241013</td>
      <td>0.035277</td>
      <td>0.142637</td>
      <td>-0.379090</td>
      <td>0.332115</td>
      <td>0.149918</td>
      <td>0.208133</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#상관계수 행렬
plt.figure(figsize=(10,10))
sns.heatmap(data = df_x.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2f526b225c0>




![png](week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_35_1.png)



```python
#VIF확인하기
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    df_x.values, i) for i in range(df_x.shape[1])]
vif["features"] = df_x.columns
vif.sort_values(["VIF Factor"], ascending=[False])

#s1, s2, s3, s5의 vif가 10이상이므로 다중공선성이 있다고 판단된다.
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>59.203786</td>
      <td>s1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39.194379</td>
      <td>s2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.402352</td>
      <td>s3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.076222</td>
      <td>s5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.890986</td>
      <td>s4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.509446</td>
      <td>bmi</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.484623</td>
      <td>s6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.459429</td>
      <td>bp</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.217307</td>
      <td>age</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.147844</td>
      <td>sex1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.130229</td>
      <td>sex0</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
    다중공선성 의심 변수: s1, s2, s3, s5
    이 변수들을 삭제할 수도 있고, 일부만 제거할 수도 있음
"""
```




    '\n    다중공선성 의심 변수: s1, s2, s3, s5\n    이 변수들을 삭제할 수도 있고, 일부만 제거할 수도 있음\n'



# c) Modeling


```python
# train, test data 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
```


```python
from sklearn.linear_model import LinearRegression

#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
#fit된 모델의 R-square
model.score(X_train, y_train)
```




    0.5533691593009212




```python
#MSE
import sklearn as sk
sk.metrics.mean_squared_error(y_train, model.predict(X_train))
```




    2738.158640226629




```python
print(model.coef_) #추정된 회귀계수(intercept제외)
print(model.intercept_) #intercept
```

    [[-2.88025896e+01  5.56173500e+02  3.14430965e+02 -6.83830055e+02
       3.43051200e+02  2.32410114e+01  1.73770532e+02  7.33392363e+02
       4.67661674e+01  4.91755869e+15  4.91755869e+15]]
    [-4.91755869e+15]



```python
#test데이터 예측
model.predict(X_test)
```




    array([[238.],
           [251.],
           [164.],
           [123.],
           [190.],
           [261.],
           [116.],
           [188.],
           [151.],
           [238.],
           [175.],
           [178.],
           [108.],
           [ 94.],
           [244.],
           [ 86.],
           [159.],
           [ 69.],
           [103.],
           [222.],
           [196.],
           [163.],
           [161.],
           [158.],
           [196.],
           [170.],
           [123.],
           [ 87.],
           [195.],
           [160.],
           [175.],
           [ 87.],
           [148.],
           [148.],
           [144.],
           [200.],
           [169.],
           [191.],
           [131.],
           [209.],
           [ 87.],
           [163.],
           [146.],
           [187.],
           [178.],
           [ 77.],
           [145.],
           [141.],
           [120.],
           [238.],
           [162.],
           [ 73.],
           [158.],
           [156.],
           [240.],
           [174.],
           [193.],
           [120.],
           [136.],
           [168.],
           [217.],
           [174.],
           [160.],
           [108.],
           [259.],
           [153.],
           [ 85.],
           [235.],
           [201.],
           [ 48.],
           [ 80.],
           [128.],
           [103.],
           [146.],
           [131.],
           [190.],
           [ 97.],
           [196.],
           [219.],
           [188.],
           [152.],
           [208.],
           [ 43.],
           [206.],
           [ 75.],
           [ 98.],
           [146.],
           [193.],
           [135.]])




```python
#test데이터 R-square
model.score(X_test, y_test)
```




    0.33193540056196336




```python
# 예측 vs. 실제데이터 plot
y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')

# 예측과 실제가 비슷하면, 라인상에 분포함
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
```


![png](week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk1_%EA%B8%B8%ED%83%9C%ED%98%95_46_0.png)



```python
"""
    MSE: 2738
    train R-square: 0.55
    test R-square: 0.33
"""
```




    '\n    MSE: 2738\n    train R-square: 0.55\n    test R-square: 0.33\n'




```python
#다중공선성이 제일 큰 변수를 제거하고 다시하기
df2 = df.drop(['s1'],axis=1)
df2.head()
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
      <th>sex0</th>
      <th>sex1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>206.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>135.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2_x = df2.drop(['target'], axis=1)
df2_y = pd.DataFrame(df2['target'],columns=['target'])
df2_x.head()
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>sex0</th>
      <th>sex1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# train, test data 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2_x, df2_y, test_size=0.2, random_state=0)
```


```python
#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(X_train, y_train)
#fit된 모델의 R-square
model.score(X_train, y_train)
```




    0.5509511284994073




```python
#MSE
import sklearn as sk
sk.metrics.mean_squared_error(y_train, model.predict(X_train))
```




    2752.982855950595




```python
#test데이터 R-square
model.score(X_test, y_test)
```




    0.3235080332827154




```python
"""
    MSE: 2752
    train R-square: 0.55
    test R-square: 0.32
"""
```




    '\n    MSE: 3229 (증가)\n    train R-square: 0.47 (감소)\n    test R-square: 0.32 (거의비슷, 약간감소)\n'




```python
"""
    다중공선성이 가장 큰 S1변수를 제거하고 회귀한 결과,
    기존의 MSE, train R-square, test R-square과 거의 비슷하므로
    굳이 이 변수를 사용할 필요가 없다.
    
    하지만 모델의 accuracy를 높이는게 가장 큰 목적이면
    이 변수도 사용해서 정확도를 높이는게 좋다.
"""
```




    '\n    다중공선성이 가장 큰 S1변수를 제거하고 회귀한 결과,\n    기존의 MSE, train R-square, test R-square과 거의 비슷하므로\n    굳이 이 변수를 사용할 필요가 없다.\n    \n    하지만 모델의 accuracy를 높이는게 가장 큰 목적이면\n    이 변수도 사용해서 정확도를 높이는게 좋다.\n'




```python
#Ridge, Lasso 회귀
from sklearn.linear_model import Ridge, Lasso

ridge=Ridge(alpha=1.0)#alpha: 얼마나 정규화를 할건지 정하는 양수 하이퍼파라미터 (클수록 더 정규화)
ridge.fit(X_train, y_train)
```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
ridge.get_params
```




    <bound method BaseEstimator.get_params of Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)>




```python
#R-square
ridge.score(X_train,y_train)
```




    0.4724942595995212




```python
#정규화를 덜하니까 R-square가 오히려 증가했다.
ridge=Ridge(alpha=0.3)
ridge.fit(X_train, y_train)
ridge.score(X_train,y_train)
```




    0.5337782773542921




```python
#Lasso
lasso=Lasso(alpha=0.3)
lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
```




    0.5275861106431708




```python
"""
    정규화를 많이하니까 오히려 R-square가 감소했다.
    overfitting의 문제는 아니고, 모델이 단순해서 설명력이 부족한 것같다.
    더 복잡한 모델(다항회귀, DT 등)이 필요해보인다.
"""
```




    '\n    정규화를 많이하니까 오히려 R-square가 감소했다.\n    overfitting의 문제는 아니고, 모델이 단순해서 설명력이 부족한 것같다.\n    더 복잡한 모델(다항회귀, DT 등)이 필요해보인다.\n'




```python

```


```python
"""
    <reference>
    https://www.kaggle.com/andyxie/beginner-scikit-learn-linear-regression-tutorial
"""
```
