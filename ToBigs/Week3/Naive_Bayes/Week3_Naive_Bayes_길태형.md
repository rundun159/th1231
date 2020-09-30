# German Credit Dataset
- 대출인지 아닌지를 예측하는 문제
- 데이터를 NB에 맞도록 간단하게 변환합니다.
- Binary 데이터들로 이루어진 대출 사기 데이터들로 부터 대출인지 아닌지 예측해보세요.


```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
```


```python
data_url = './fraud.csv'
df = pd.read_csv(data_url, sep=',')
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
      <th>ID</th>
      <th>History</th>
      <th>CoApplicant</th>
      <th>Accommodation</th>
      <th>Fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>current</td>
      <td>none</td>
      <td>own</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>paid</td>
      <td>none</td>
      <td>own</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>paid</td>
      <td>none</td>
      <td>own</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>paid</td>
      <td>guarantor</td>
      <td>rent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>arrears</td>
      <td>none</td>
      <td>own</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID열을 삭제해줍니다.
del df["ID"] 

# Label(Y_data)을 따로 저장해 줍니다.
Y_data = df.pop("Fraud")
```


```python
# as_matrix()함수를 통해 array형태로 변환시켜 줍니다.
# Convert the frame to its Numpy-array representation.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as_matrix.html

Y_data = Y_data.as_matrix()
Y_data
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\ipykernel_launcher.py:5: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """





    array([ True, False, False,  True, False,  True, False, False, False,
            True, False,  True,  True, False, False, False, False, False,
           False, False])




```python
type(Y_data)
```




    numpy.ndarray




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
      <th>History</th>
      <th>CoApplicant</th>
      <th>Accommodation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>current</td>
      <td>none</td>
      <td>own</td>
    </tr>
    <tr>
      <th>1</th>
      <td>paid</td>
      <td>none</td>
      <td>own</td>
    </tr>
    <tr>
      <th>2</th>
      <td>paid</td>
      <td>none</td>
      <td>own</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paid</td>
      <td>guarantor</td>
      <td>rent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arrears</td>
      <td>none</td>
      <td>own</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 우리가 앞으로 사용할 데이터 셋입니다. 그런데 문제가 있어보이네요...
```

## One-Hot encoding

* 범주형 변수를 dummy변수로 변환해주는 작업
* Do it yourself!

### 1. Do One-Hot Encoding! 


```python
df.columns
```




    Index(['History', 'CoApplicant', 'Accommodation'], dtype='object')




```python
df.isnull().sum()
```




    History          0
    CoApplicant      0
    Accommodation    0
    dtype: int64




```python
# 범주형 변수 처리 문제입니다.
# 앞선 EDA 시간과 Logistic EDA를 통해 우리는 범주형 변수를 처리해 주는 방법을 배웠습니다.
# get_dummies를 사용해서 One-Hot encoding 처리를 해주세요.

```


```python
df['History'].unique()
```




    array(['current', 'paid', 'arrears', 'none'], dtype=object)




```python
df['CoApplicant'].unique()
```




    array(['none', 'guarantor', 'coapplicant'], dtype=object)




```python
df['Accommodation'].unique()
```




    array(['own', 'rent', 'free'], dtype=object)



- none값이 History feature에 등장하고, CoApplicant에도 등장하므로 
- none값을 각각 History_none, CoApplicant_none으로 변경하려고 합니다.


```python
df.loc[df['History']=='none','History']='History_none'
```


```python
df.loc[df['CoApplicant']=='none','CoApplicant']='CoApplicant_none'
```


```python
dummy_var = pd.get_dummies(df.Accommodation)
del df['Accommodation']
df = pd.concat([df,dummy_var], axis=1)
# pd.concat([df.drop(['Accommodation'], axis=1),dummy_var], axis=1).head()
```


```python
dummy_var = pd.get_dummies(df.History)
del df['History']
df = pd.concat([df,dummy_var], axis=1)
# pd.concat([df.drop(['History'], axis=1),dummy_var], axis=1).head()
```


```python
dummy_var = pd.get_dummies(df.CoApplicant)
del df['CoApplicant']
df = pd.concat([df,dummy_var], axis=1)
# pd.concat([df.drop(['History'], axis=1),dummy_var], axis=1).head()
```


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
      <th>free</th>
      <th>own</th>
      <th>rent</th>
      <th>History_none</th>
      <th>arrears</th>
      <th>current</th>
      <th>paid</th>
      <th>CoApplicant_none</th>
      <th>coapplicant</th>
      <th>guarantor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* One-Hot Encoding이 제대로 되었다면 우리는 10개의 Feature를 얻을 수 있습니다.


```python
x_data = df.as_matrix()
x_data
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.





    array([[0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=uint8)




```python
# one-hot encoding을 통해 10개의 Feature를 얻었다.
```

#### Q1. as_matrix()함수를 통해 우리가 하고자 하는 것은 무엇일까요? 

- x_data는 type이 pandas의 DataFrame인데, as_matrix를 통해, 계산이 용이한 numpy array로 변환합니다.


```python
Y_data == True # boolean index
```




    array([ True, False, False,  True, False,  True, False, False, False,
            True, False,  True,  True, False, False, False, False, False,
           False, False])




```python
len(set(Y_data))
```




    2



    대출 할지 말지 예측하는게 포인트

## Naive bayes classifier

* P(Y)
* P(X1, X2, ..., Xn)
* P(Y|X1, X2, X3, ..., Xn)
* P(X1|Y), P(X2|Y), ... P(Xn|Y)
등 우리가 구해야 할 식들에 대한 아이디어가 있어야 합니다.

### P(Y1), P(Y0) 구하기


```python
# P(Y1), P(Y0)
# P(Y1) = count(Y1) / count(Y)

P_Y_True = sum(Y_data==True) / len(Y_data)
P_Y_False = 1 - P_Y_True

P_Y_True, P_Y_False
```




    (0.3, 0.7)



* 이번 튜토리얼에서는 **index를 이용합니다.**
* 이해하기보다는 따라 하면서 음미해보세요.


```python
# y가 1일 경우, y가 0일 경우를 구해줘야 합니다.
# 이번 시간에는 np.where를 사용합니다.
# np.where
```


```python
ix_Y_True = np.where(Y_data) # Y_data == True인 인덱스를 뽑아줍니다.
ix_Y_False = np.where(Y_data==False)

ix_Y_True, ix_Y_False
```




    ((array([ 0,  3,  5,  9, 11, 12], dtype=int32),),
     (array([ 1,  2,  4,  6,  7,  8, 10, 13, 14, 15, 16, 17, 18, 19],
            dtype=int32),))




```python
# np.where을 사용해서 Y가1일 때와 0일 때 각각의 인덱스 값을 얻을 수 있게 되었습니다.
```


```python
# P(X|Y) = count(X_cap_Y) / count(Y)
```

### productP(X|Yc) 구하기

* product * P(X|Y1)
* product * P(X|Y2)


```python
p_x_y_true = x_data[ix_Y_True].sum(axis=0) / sum(Y_data == True)
p_x_y_true
```




    array([0.        , 0.66666667, 0.33333333, 0.16666667, 0.16666667,
           0.5       , 0.16666667, 0.83333333, 0.        , 0.16666667])




```python
p_x_y_true = (x_data[ix_Y_True].sum(axis=0) / sum(Y_data==True))  # Q.뒤에 sum(Y_data == True) 필요한가요? # 앞에 식이 P(X_cap_Y1)인 것 같은데...
p_x_y_false = (x_data[ix_Y_False].sum(axis=0) / sum(Y_data==False))

p_x_y_true, p_x_y_false
```




    (array([0.        , 0.66666667, 0.33333333, 0.16666667, 0.16666667,
            0.5       , 0.16666667, 0.83333333, 0.        , 0.16666667]),
     array([0.07142857, 0.78571429, 0.14285714, 0.        , 0.42857143,
            0.28571429, 0.28571429, 0.85714286, 0.14285714, 0.        ]))



- Q.뒤에 sum(Y_data == True) 필요한가요?
- P(X|Y1)을 구하기 위해서는, P(X_cap_Y1)을 P(Y1)로 나눠야 하기 때문에, 
- sum(Y_data==True)가 분모에 있게 됩니다.


```python
# 총 10개의 값에 대해서 확률을 구해준다.
```


```python
x_test = [0,1,0,0,0,1,0, 0,1,0]


import math

p_y_true_test = P_Y_True * p_x_y_true.dot(x_test)
p_y_false_test = P_Y_False * p_x_y_false.dot(x_test)

p_y_true_test, p_y_false_test
```




    (0.3499999999999999, 0.8499999999999999)




```python
p_y_true_test < p_y_false_test
```




    True



## 2. Do Smoothing을 통해 P(Y=1|X)의 확률과 P(Y=0|X)의 확률 값을 비교하세요.


```python
smoothing_p = 2
```


```python
def get_p_cond_x_y(x_data,y_data,x,p):   #각 feature x에 대해, P(X|Y)를 구하는 함수. p:는 smpoothing_p
    numofF = len(x)
    result=np.empty(numofF,np.float32)
    num_class_1 = y_data.sum()          #y가 참인 클래스에 속하는 경우의 수 
    num_class_0 = len(y_data)-y_data.sum()   #y가 거짓인 클래스에 속하는 경우의 수 
    for i in range(0,numofF):
        divisor=p*numofF                #스무딩 적용. 분자+=p*(클래스의 갯수)
        dividend=p                      #스무딩 적용. 분모+=p
        if(x_test[i]):                  #x_data[i]의 값이 참일 경우
            divisor+=num_class_1
            dividend+=x_data[y_data][:,i].sum()
        else:
            divisor+=num_class_0
            dividend+=len(x_data)-x_data[y_data][:,i].sum()
        result[i]=dividend/divisor
    return result
```

- 각 feature에 대한 조건부확률을 구하는 함수입니다.
- 스무딩을 적용하기 위해, divisor와 dividend를 스무딩 계수와 클래스 수를 이용해 초기화 했고,
- count(X cap Y)/count(Y)값 (스무딩을 제외하고 쓴다면)
- 으로 계산했습니다.


```python
log_p_y_true_test=math.log(P_Y_True)
log_p_y_false_test=math.log(P_Y_False)

import math
Y_False_data = ~np.bool_(Y_data)
log_p_y_true_test+=np.log(get_p_cond_x_y(x_data,Y_data,x_test,2)).sum()
log_p_y_false_test+=np.log(get_p_cond_x_y(x_data,Y_False_data,x_test,2)).sum()
```


```python
log_p_y_true_test < log_p_y_false_test
```




    True




```python
log_p_y_true_test
```




    -8.11961029745948




```python
log_p_y_false_test
```




    -5.382908617034436



- 로그값을 안취하고 계산해봤더니 0이 나와서 두 값을 로그값을 취해서 비교했습니다.

- classifiying에 필요한 건, 정확한 값이 아닌 두 값 사이의 대소비교이므로,
- 두 값 모두 분모에 있는 P(X)는 계산하지 않았습니다. 


```python
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns    
plt.figure(figsize=(10,10))
sns.heatmap(data = df.corr(), annot=True, 
fmt = '.2f', linewidths=.3, cmap='Blues')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xbd18510>




![png](Week3_Naive_Bayes_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week3_Naive_Bayes_%EA%B8%B8%ED%83%9C%ED%98%95_58_1.png)


- Naive_Bayes를 적용할때, 각 feature간 독립이라는 가정이 있었습니다.
- correlation을 확인해보니 rent와 own이 0.87만큼, coapllicant와 coApplicant_none이 0.79만큼 관련이 있었습니다. 
- 아무래도 rent와 own, coapplicant와 coApplicant_none이 본래 한 feature의 값이었으므로, 
- 상관관계가 있는 것은 당연한 결과일 것입니다.

- 이 값들을 모두 포함시키고 계산했던 위의 결과는 부정확할 수 있다고 생각하지만, 
- log_p_y_true_test, log_p_y_false_test 모두 같은 계산 방식을 선택했으므로,
- 즉, 두 계산 과정 모두 상관관계가 없는것으로 해석했으므로, 
- 두 계산 결과의 대소결과는 차이가 없을것이라고 예상합니다. 


```python
df2=df.drop(['rent','CoApplicant_none'],axis=1)
```


```python
x_data_2=df2.as_matrix()
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.



```python
x_test2=[0,1,0,0,1,0,1,0]    #rent와 coApplicant_none의 데이터 제거
```


```python
log_p_y_true_test=math.log(P_Y_True)
log_p_y_false_test=math.log(P_Y_False)

import math
Y_False_data = ~np.bool_(Y_data)
log_p_y_true_test+=np.log(get_p_cond_x_y(x_data_2,Y_data,x_test2,2)).sum()
log_p_y_false_test+=np.log(get_p_cond_x_y(x_data_2,Y_False_data,x_test2,2)).sum()
```


```python
log_p_y_true_test
```




    -8.11961029745948




```python
log_p_y_false_test
```




    -5.382908617034436




```python
log_p_y_true_test < log_p_y_false_test
```




    True



- 두 값 모두 큰 차이가 없고,
- 대소 비교 결과는 변함이 없습니다.

# 결과값에 대한 설명과 해석을 달아주세요
