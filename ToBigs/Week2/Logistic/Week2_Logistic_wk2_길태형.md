# Money Ball - Basic EDA & Modeling(Logistic)

### 2주차 과제는 Money Ball Data Analysis입니다. 
### 우리가 Binary Classification하고자 하는 변수는 play-off입니다.
### Money Ball Data Set을 분석하고 어느 팀이 play-off에 진출하는지 
### Logistic Regression방법을 통해 분석합니다.
_How does a team make the playoffs?_

_How does a team win more games?_

_How does a team score more runs?_

- 머니볼과 빅데이터
- http://writting.co.kr/2015/04/%EB%A8%B8%EB%8B%88%EB%B3%BC-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0/

- _이번 과제를 통해 우리는 어떤 팀이  play-off(가을야구)에 진출하는지 로지스틱 모델을 통해 분석합니다._
+ _Ipython 파일의 빈 부분을 채워주세요._
+ _하나하나 천천히 따라와 주세요._ 

+ _W(Wins) Feature를 제외한 Feature들 중에서 가을야구 진출에 가장 영향을 많이 주는 Feature는 무엇일까요?_
+ _통념이 만연하던 야구의 편견을 깬 Money Ball은 과연 무엇일까요?_

### What is Money Ball?
- Billy Bean & DePodesta's Story.

In the early 2000s, Billy Beane and Paul DePodesta worked for the Oakland Athletics. While there, they literally changed the game of baseball. They didn't do it using a bat or glove, and they certainly didn't do it by throwing money at the issue; in fact, money was the issue. They didn't have enough of it, but they were still expected to keep up with teams that had much deeper pockets. This is where Statistics came riding down the hillside on a white horse to save the day. This data set contains some of the information that was available to Beane and DePodesta in the early 2000s, and it can be used to better understand their methods.

# 1. Import Library

- 앞으로 데이터 분석 및 모델링을 함에 있어서 첫 스텝은 필요한 Library를 불러오는 것입니다.


```python
# python library import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../EDA_MoneyBall/"))
print(os.listdir("./"))
```

    ['baseball.csv']
    ['.ipynb_checkpoints', 'baseball.csv', 'Logistic과제공지.pdf', 'Logistic과제공지.pptx', 'MoneyBall_EDA_Logistic_Help_org.ipynb', 'Moneyball_Explanation.ipynb', 'Week2_Logistic_wk2_길태형.ipynb']



```python
from sklearn.linear_model import LogisticRegression # sklearn을 사용하여 Logistic 회귀분석을 할 경우 필요
import matplotlib.pyplot as plt # 시각화를 위한 library
import warnings
warnings.filterwarnings('ignore')

## Jupyter Notebook 이나 ipython 을 사용하다보면 향후 버전이 올라갈 때 변경될 사항 등을 알려주는 경고 메시지(warning message)가 거슬릴 때가 있습니다.
## 출처: https://rfriend.tistory.com/346 [R, Python 분석과 프로그래밍 (by R Friend)]
```

# 2. Load Data & Data Exploration 

* dataset 불러오기 
* pandas를 이용해서 CSV파일을 불러오세요
* 불러온 데이터를 파악해봅니다


```python
data = pd.read_csv("./baseball.csv")
data.head() # pandas로 data를 불러오면 습관적으로 head()를 찍어봅니다!
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
      <th>Team</th>
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>W</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARI</td>
      <td>NL</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>81</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATL</td>
      <td>NL</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>94</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAL</td>
      <td>AL</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>93</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BOS</td>
      <td>AL</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>69</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHC</td>
      <td>NL</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>61</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Year'].unique()
```




    array([2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002,
           2001, 2000, 1999, 1998, 1997, 1996, 1993, 1992, 1991, 1990, 1989,
           1988, 1987, 1986, 1985, 1984, 1983, 1982, 1980, 1979, 1978, 1977,
           1976, 1975, 1974, 1973, 1971, 1970, 1969, 1968, 1967, 1966, 1965,
           1964, 1963, 1962], dtype=int64)




```python
# 주어진 데이터 셋이 어떤 데이터인지 한번 쭉 살펴봅니다.
```

* Tip. 우리가 일반적으로 알고 있는 데이터 셋이 아닐 경우 각각의 Feature(Column, Attribute)가 무엇을 의미하는지 이해할 필요가 있습니다.
**우리는 앞으로 Feature로 통일하겠습니다**
- Data Set에 대한 설명을 참조합니다.
- https://www.kaggle.com/wduckett/moneyball-mlb-stats-19622012

## 2-1. 각각의 데이터가 무엇을 의미하는지 파악하기
    - 이미 알고 있는 내용이라면 건너 뛰셔도 됩니다.
    e.g) Team: Major League Team 이름이구나.
         League: 소속 League를 말하는구나.
         Year: 데이터가 기록된 년도를 의미하는구나.
         Rs: (Runs Scored) 득점 스코어를 의미하는구나
         RA: (Runs Allowed) 실점스코어를 의미하는구나
         .
         .
         .


```python
# 어떤 Feature가 있을까?
data.columns
```




    Index(['Team', 'League', 'Year', 'RS', 'RA', 'W', 'OBP', 'SLG', 'BA',
           'Playoffs', 'RankSeason', 'RankPlayoffs', 'G', 'OOBP', 'OSLG'],
          dtype='object')




```python
# Feature는 몇 개일까?
len(data.columns)
```




    15



* 특정 feature는 종속 변수에 아무런 영향을 주지 않을 수 있습니다. 
* 그런 feature들을 파악하고 제거한다면 우리의 모델은 더욱 정확해 집니다.
* 하지만 마냥 변수를 제거할 수도 없는 노릇입니다. 
* 지난 1주차에 배웠던 EDA와 Preprocessing을 참고하여 데이터를 분석해봅시다.

## Data Exploration


```python
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 각종 시각화 패키지 불러오기
```


```python
# for using Korean font
matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False
```


```python
display(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1232 entries, 0 to 1231
    Data columns (total 15 columns):
    Team            1232 non-null object
    League          1232 non-null object
    Year            1232 non-null int64
    RS              1232 non-null int64
    RA              1232 non-null int64
    W               1232 non-null int64
    OBP             1232 non-null float64
    SLG             1232 non-null float64
    BA              1232 non-null float64
    Playoffs        1232 non-null int64
    RankSeason      244 non-null float64
    RankPlayoffs    244 non-null float64
    G               1232 non-null int64
    OOBP            420 non-null float64
    OSLG            420 non-null float64
    dtypes: float64(7), int64(6), object(2)
    memory usage: 134.8+ KB



    None



```python
# 1232개의 entries, 15개의 column
# null 값이 있는지 isnull.sum()으로 확인을 해보자.
```


```python
data.isnull().sum()
```




    Team              0
    League            0
    Year              0
    RS                0
    RA                0
    W                 0
    OBP               0
    SLG               0
    BA                0
    Playoffs          0
    RankSeason      988
    RankPlayoffs    988
    G                 0
    OOBP            812
    OSLG            812
    dtype: int64




```python
# 어떻게 처리할지 생각해보자
```

# Q. Null 값을 어떻게 전처리 할 것인가?
- 데이터를 본격적으로 분석 하기 전에 한번 생각해보도록 합니다.


```python
data.shape[0]
```




    1232




```python
data.shape[1]
```




    15




```python
data['RankSeason'].unique()
```




    array([nan,  4.,  5.,  2.,  6.,  3.,  1.,  7.,  8.])



- RankPlayoffs, RankSeason 처리
- 두 feature은 팀의 성적과 관련된 것 같습니다.


```python
data['RankSeason'].unique()
```




    array([nan,  4.,  5.,  2.,  6.,  3.,  1.,  7.,  8.])



- PlayOffs에 진출하지 못한 시즌은 두 feature 값이 null인것을 알 수 있습니다.
- 두 feature값의 null 값을 0으로 바꾸겠습니다.


```python
data.loc[data['Playoffs']==float(0.0),'RankPlayoffs']=0
```


```python
data.loc[data['Playoffs']==float(0.0),'RankSeason']=0
```


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
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>W</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1988.957792</td>
      <td>715.081981</td>
      <td>715.081981</td>
      <td>80.904221</td>
      <td>0.326331</td>
      <td>0.397342</td>
      <td>0.259273</td>
      <td>0.198052</td>
      <td>0.618506</td>
      <td>0.538149</td>
      <td>161.918831</td>
      <td>0.332264</td>
      <td>0.419743</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.819625</td>
      <td>91.534294</td>
      <td>93.079933</td>
      <td>11.458139</td>
      <td>0.015013</td>
      <td>0.033267</td>
      <td>0.012907</td>
      <td>0.398693</td>
      <td>1.465193</td>
      <td>1.187604</td>
      <td>0.624365</td>
      <td>0.015295</td>
      <td>0.026510</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1962.000000</td>
      <td>463.000000</td>
      <td>472.000000</td>
      <td>40.000000</td>
      <td>0.277000</td>
      <td>0.301000</td>
      <td>0.214000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>158.000000</td>
      <td>0.294000</td>
      <td>0.346000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1976.750000</td>
      <td>652.000000</td>
      <td>649.750000</td>
      <td>73.000000</td>
      <td>0.317000</td>
      <td>0.375000</td>
      <td>0.251000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.321000</td>
      <td>0.401000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1989.000000</td>
      <td>711.000000</td>
      <td>709.000000</td>
      <td>81.000000</td>
      <td>0.326000</td>
      <td>0.396000</td>
      <td>0.260000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.331000</td>
      <td>0.419000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2002.000000</td>
      <td>775.000000</td>
      <td>774.250000</td>
      <td>89.000000</td>
      <td>0.337000</td>
      <td>0.421000</td>
      <td>0.268000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.343000</td>
      <td>0.438000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2012.000000</td>
      <td>1009.000000</td>
      <td>1103.000000</td>
      <td>116.000000</td>
      <td>0.373000</td>
      <td>0.491000</td>
      <td>0.294000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>165.000000</td>
      <td>0.384000</td>
      <td>0.499000</td>
    </tr>
  </tbody>
</table>
</div>



- 'OOBP', 'OSLG' 이 null값인 데이터의 비율이 25%정도입니다.
- Feature를 삭제하기엔 애매하고, OOBP가 누락된 데이터에 특징이 보이지 않아서,
- 각 Feature의 평균값을 결측치 대신에 대입하려고 합니다.


```python
data.loc[data['OOBP'].isnull(),'OOBP']=data['OOBP'].mean()
```


```python
data.loc[data['OSLG'].isnull(),'OSLG']=data['OSLG'].mean()
```

- 이산형 변수 'W', normalizing


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
w = np.array(data['W']).reshape(-1,1)
scaler.fit(w)
data['W']=scaler.transform(w)
```


```python
data['W'].describe()
```




    count    1.232000e+03
    mean     2.270911e-17
    std      1.000406e+00
    min     -3.571333e+00
    25%     -6.901147e-01
    50%      8.362450e-03
    75%      7.068396e-01
    max      3.064200e+00
    Name: W, dtype: float64



## 2-2.  변수 종류 확인하기


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1232 entries, 0 to 1231
    Data columns (total 15 columns):
    Team            1232 non-null object
    League          1232 non-null object
    Year            1232 non-null int64
    RS              1232 non-null int64
    RA              1232 non-null int64
    W               1232 non-null float64
    OBP             1232 non-null float64
    SLG             1232 non-null float64
    BA              1232 non-null float64
    Playoffs        1232 non-null int64
    RankSeason      1232 non-null float64
    RankPlayoffs    1232 non-null float64
    G               1232 non-null int64
    OOBP            1232 non-null float64
    OSLG            1232 non-null float64
    dtypes: float64(8), int64(5), object(2)
    memory usage: 134.8+ KB


###  1) 범주형 변수 확인하기


```python
# categorical variable
categorical_col = list(data.select_dtypes(include='object').columns)
categorical_col
```




    ['Team', 'League']




```python
# play-off Feature의 경우 0과 1로 범주형이지만 데이터에는 int type로 저장되어있다.
```

###  2) 연속형 변수 확인하기


```python
# numerical variable
numerical_col =  list(data.select_dtypes(include=('int64', 'float64')).columns)
numerical_col
```




    ['Year',
     'RS',
     'RA',
     'W',
     'OBP',
     'SLG',
     'BA',
     'Playoffs',
     'RankSeason',
     'RankPlayoffs',
     'G',
     'OOBP',
     'OSLG']




```python
len(numerical_col)
```




    13



### 3) 변수 종류 확인
- 2개의 categorical variable(Team, League)와 13개의 numerical variable

###  4) 각각의 변수에 어떤 값이 들어있을까?
- 각 변수별 unique값을 찍어본다.


```python
# for categorical_col

for col in categorical_col:
    print(col + ': ', len(set(data[str(col)])))
```

    Team:  39
    League:  2



```python
# 39개의 팀과 2개의 리그
# 리그는 지난 수십년간 2개였다.
```


```python
data['Year'].unique()
```




    array([2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002,
           2001, 2000, 1999, 1998, 1997, 1996, 1993, 1992, 1991, 1990, 1989,
           1988, 1987, 1986, 1985, 1984, 1983, 1982, 1980, 1979, 1978, 1977,
           1976, 1975, 1974, 1973, 1971, 1970, 1969, 1968, 1967, 1966, 1965,
           1964, 1963, 1962], dtype=int64)




```python
# for numerical_col

for col in numerical_col:
    print(col + ': ', len(set(data[str(col)])))
```

    Year:  47
    RS:  374
    RA:  381
    W:  63
    OBP:  87
    SLG:  162
    BA:  75
    Playoffs:  2
    RankSeason:  9
    RankPlayoffs:  6
    G:  8
    OOBP:  73
    OSLG:  113



```python
# 지난 수십년은 46년이었다.
# Playoffs가 2인 것으로 보아 categorical 변수로 바꿔줘도 무관할 것 같다.
# Q. G(Games Played)는 어떤 값을 가지고 있을까?
```


```python
data.G.head()
```




    0    162
    1    162
    2    162
    3    162
    4    162
    Name: G, dtype: int64




```python
# Game 수이다. 한 시즌에 치뤄진 경기수를 나타낸다.
```


```python
data.G.mean()
```




    161.91883116883116




```python
# 47년간 평균 161.918경기가 치뤄졌다. 
```

# 2. Data Preprocessing 

### T1. Column 삭제하기
- W(Wins), 승리 외에 팀의 가을야구 진출에 영향을 많이 미치는 Feature가 알고 싶습니다.
- del or drop을 사용하여 W column을 삭제해주세요


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
      <th>Team</th>
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>W</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARI</td>
      <td>NL</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>0.008362</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATL</td>
      <td>NL</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>1.143388</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAL</td>
      <td>AL</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>1.056078</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BOS</td>
      <td>AL</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>-1.039353</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHC</td>
      <td>NL</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>-1.737831</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
W_data = data['W']
```


```python
del data['W']
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
      <th>Team</th>
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARI</td>
      <td>NL</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATL</td>
      <td>NL</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAL</td>
      <td>AL</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BOS</td>
      <td>AL</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHC</td>
      <td>NL</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
# W Column을 지워줬다.
# 저는 나중에 regression 모델 score 매길 때 필요할 것 같아서, W_data에 따로 저장했습니다.
```


```python
W_data
```




    0       0.008362
    1       1.143388
    2       1.056078
    3      -1.039353
    4      -1.737831
              ...   
    1227    0.008362
    1228    1.056078
    1229    1.929175
    1230    0.270291
    1231   -1.825140
    Name: W, Length: 1232, dtype: float64




```python
W_data.describe()
```




    count    1.232000e+03
    mean     2.270911e-17
    std      1.000406e+00
    min     -3.571333e+00
    25%     -6.901147e-01
    50%      8.362450e-03
    75%      7.068396e-01
    max      3.064200e+00
    Name: W, dtype: float64



- W data의 평균이 0에 근사하고, standard deviation이 1에 근사한 것을 보면,
- W data는 scaling 되어 있는 것을 알 수 있습니다.

### Task2. 인코딩: League
- League Feature는 AL과 NL로 이루어져 있습니다.
- 지난 시간에 배웠던 인코딩 방법을 적용해서 모델이 학습 할 수 있도록 처리해주세요.
- replace()함수를 사용합니다.


```python
set(data.League)
```




    {'AL', 'NL'}




```python
data.League.replace({'AL':0, 'NL':1}, inplace=True)
```


```python
data.League.head()
```




    0    1
    1    1
    2    0
    3    0
    4    1
    Name: League, dtype: int64



### Task3. column 삭제하기
- Team column을 삭제해주세요.
- Team column이 없어도 모델에는 큰 영향이 없을 것 같습니다.


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
      <th>Team</th>
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARI</td>
      <td>1</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATL</td>
      <td>1</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAL</td>
      <td>0</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BOS</td>
      <td>0</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHC</td>
      <td>1</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
del data['Team']
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
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>



### Task4. NaN값 처리하기
- head를 찍어보니 NaN값이 보입니다. 
- NaN값을 처리해 줍니다.


```python
data.isnull().sum()
```




    League          0
    Year            0
    RS              0
    RA              0
    OBP             0
    SLG             0
    BA              0
    Playoffs        0
    RankSeason      0
    RankPlayoffs    0
    G               0
    OOBP            0
    OSLG            0
    dtype: int64




```python
# RankSeason, RankPlayoffs, OOBP, OSLG 변수에 Null 값이 있습니다.
# 우리는 위의 변수들에 NaN값을 처리해줄 것입니다.
```

data.RankSeason.head()


```python
data.RankSeason.head()
```




    0    0.0
    1    4.0
    2    5.0
    3    0.0
    4    0.0
    Name: RankSeason, dtype: float64




```python
data.RankPlayoffs.head()
```




    0    0.0
    1    5.0
    2    4.0
    3    0.0
    4    0.0
    Name: RankPlayoffs, dtype: float64



- 태형 : 어...저는 위쪽에서 RankSeason이랑 RankPlayoffs 처리하고 왔습니당...ㅎㅎㅎ


```python
# OOBP는 Opponent On-Base Percentage. 
# OSLG는 Opponent Slugging Percentage.

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
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>RankSeason</th>
      <th>RankPlayoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
      <td>1232.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>1988.957792</td>
      <td>715.081981</td>
      <td>715.081981</td>
      <td>0.326331</td>
      <td>0.397342</td>
      <td>0.259273</td>
      <td>0.198052</td>
      <td>0.618506</td>
      <td>0.538149</td>
      <td>161.918831</td>
      <td>0.332264</td>
      <td>0.419743</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500203</td>
      <td>14.819625</td>
      <td>91.534294</td>
      <td>93.079933</td>
      <td>0.015013</td>
      <td>0.033267</td>
      <td>0.012907</td>
      <td>0.398693</td>
      <td>1.465193</td>
      <td>1.187604</td>
      <td>0.624365</td>
      <td>0.008924</td>
      <td>0.015466</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1962.000000</td>
      <td>463.000000</td>
      <td>472.000000</td>
      <td>0.277000</td>
      <td>0.301000</td>
      <td>0.214000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>158.000000</td>
      <td>0.294000</td>
      <td>0.346000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1976.750000</td>
      <td>652.000000</td>
      <td>649.750000</td>
      <td>0.317000</td>
      <td>0.375000</td>
      <td>0.251000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.332264</td>
      <td>0.419743</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>1989.000000</td>
      <td>711.000000</td>
      <td>709.000000</td>
      <td>0.326000</td>
      <td>0.396000</td>
      <td>0.260000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.332264</td>
      <td>0.419743</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2002.000000</td>
      <td>775.000000</td>
      <td>774.250000</td>
      <td>0.337000</td>
      <td>0.421000</td>
      <td>0.268000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.000000</td>
      <td>0.332264</td>
      <td>0.419743</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>2012.000000</td>
      <td>1009.000000</td>
      <td>1103.000000</td>
      <td>0.373000</td>
      <td>0.491000</td>
      <td>0.294000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>165.000000</td>
      <td>0.384000</td>
      <td>0.499000</td>
    </tr>
  </tbody>
</table>
</div>



* OOBP와 OSLG값에 평균 값을 넣어주도록 합니다.
* sklearndml SimpleImputer를 사용합니다.
* 자세한 내용은 https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

- 태형 : 저는 위에서 무식하게 평균 따로 넣어줘서...넘어가겠습니다


```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)
imputer = imputer.fit(data[['OOBP', 'OSLG']])

data[['OOBP', 'OSLG']] = imputer.transform(data[['OOBP', 'OSLG']])
```


```python
data.OOBP.isnull().sum()
```




    0




```python
data.OSLG.isnull().sum()
```




    0




```python
data.RankPlayoffs.isnull().sum()
```




    0




```python
data.RankSeason.isnull().sum()
```




    0




```python
del data['RankPlayoffs']
del data['RankSeason']
```


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
      <th>League</th>
      <th>Year</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2012</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2012</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2012</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2012</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
del data['Year']
```


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
      <th>League</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>



# 3. train_test_split

데이터 셋을 독립변수와 종속변수로 나눠준다.varialbe(X) and dependent(y) variable.

- Playoffs에 진출하는 확률을 추정하는 모델을 만들고 싶으므로,
- 'Playoffs' Feature를 y에 저장하려고 합니다.


```python
X = data.drop('Playoffs',axis=1)
y = data['Playoffs']
```

splitting the data set as test_set and train_set to make predictions using the calssfiers.


```python
X.head()
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
      <th>League</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Feature G를 삭제해줍니다

del X['G'] 
```


```python
X.head()
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
      <th>League</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
y
```




    0       0
    1       1
    2       1
    3       0
    4       0
           ..
    1227    0
    1228    0
    1229    1
    1230    0
    1231    0
    Name: Playoffs, Length: 1232, dtype: int64



* Splitting the data set as test_set and train_set to make predictions using the classifiers.


```python
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

# Q1. train_tset_split module
* train_test_split() 함수에 들어가는 각각의 인자 값은 무엇을 의미하는가?

- arrays
- test_size
- random_state


- 구글링 전 저의 예측
- arrays :input(독립변수,X)값과 output(종속변수,y)값을 입력 받습니다. 
- test_size : 전체 데이터를 train데이터와, test데이터로 나눌텐데, test 데이터가 전체 데이터에서 차지하는 비율을 의미할 것입니다.
- random_state : 전체 데이터에서 test 데이터를 선택하는 기준일 것 같습니다.

- 구글링해서 찾은 Ref : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- arrays : 입력으로 받는 독립 변수와 종속변수 
- lists, numpy arrays, scipy-sparse matrices, pandas dataframes의 형태로 입력 받을 수 있습니다.
- test_size : 
- float 형으로 주어질 경우 : 전체 데이터에서 train data가 차지하는 비율
- int 형으로 주어질 경우 : test data의 절대적인 수
- None일 경우 : (1-train_size)로 초기화 됩니다. default 값은 0.25
- random_state : test_data로 뽑아낼 data의 index를 random으로 생성할 generator와 관련있는 변수입니다. 
- int형으로 주어질 경우 : random number generator의 seed 값이 됩니다.
- RandomState instance로 주어질 경우 : instance가 곧바로 generator가 됩니다.
- None 일 경우 : np.random으로 주어지는 generator를 사용합니다.

# 4. Feature Scaling

경우에 따라 값이 특정 범위에서 매우 높은 범위로 변환되어 피쳐 스케일링을 사용합니다.


```python
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
warnings.filterwarnings(action='once')
```


```python
pd.DataFrame(X_test).describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
      <td>247.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.167200</td>
      <td>-0.010647</td>
      <td>-0.029802</td>
      <td>0.007054</td>
      <td>-0.061761</td>
      <td>0.011211</td>
      <td>0.058405</td>
      <td>0.058092</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.993605</td>
      <td>1.005548</td>
      <td>1.022903</td>
      <td>0.930299</td>
      <td>1.039379</td>
      <td>0.989388</td>
      <td>0.933776</td>
      <td>1.031445</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.967040</td>
      <td>-2.179515</td>
      <td>-2.629762</td>
      <td>-2.188732</td>
      <td>-2.417173</td>
      <td>-2.338232</td>
      <td>-3.115766</td>
      <td>-3.811705</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.967040</td>
      <td>-0.774097</td>
      <td>-0.730047</td>
      <td>-0.611724</td>
      <td>-0.795626</td>
      <td>-0.637343</td>
      <td>0.011709</td>
      <td>0.011647</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.034083</td>
      <td>-0.079591</td>
      <td>-0.136386</td>
      <td>-0.020346</td>
      <td>0.037879</td>
      <td>0.135789</td>
      <td>0.011709</td>
      <td>0.011647</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.034083</td>
      <td>0.647727</td>
      <td>0.565213</td>
      <td>0.636740</td>
      <td>0.659219</td>
      <td>0.676981</td>
      <td>0.011709</td>
      <td>0.011647</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.034083</td>
      <td>2.687497</td>
      <td>3.371611</td>
      <td>2.410874</td>
      <td>2.462621</td>
      <td>2.609809</td>
      <td>5.736326</td>
      <td>5.170195</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(X_train).describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
      <td>9.850000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.298454e-16</td>
      <td>-4.905270e-16</td>
      <td>-3.642884e-16</td>
      <td>-2.777248e-15</td>
      <td>-1.298454e-16</td>
      <td>5.626633e-16</td>
      <td>5.660898e-15</td>
      <td>1.052739e-15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
      <td>1.000508e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.670403e-01</td>
      <td>-2.759182e+00</td>
      <td>-2.435473e+00</td>
      <td>-3.240070e+00</td>
      <td>-2.932431e+00</td>
      <td>-3.497929e+00</td>
      <td>-4.222278e+00</td>
      <td>-4.787998e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-9.670403e-01</td>
      <td>-6.811319e-01</td>
      <td>-7.084591e-01</td>
      <td>-6.117242e-01</td>
      <td>-6.592346e-01</td>
      <td>-6.373428e-01</td>
      <td>1.170944e-02</td>
      <td>1.164661e-02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-9.670403e-01</td>
      <td>-4.677972e-02</td>
      <td>-5.003518e-02</td>
      <td>-2.034633e-02</td>
      <td>-5.304894e-02</td>
      <td>-1.883772e-02</td>
      <td>1.170944e-02</td>
      <td>1.164661e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.034083e+00</td>
      <td>6.531951e-01</td>
      <td>6.515641e-01</td>
      <td>7.024489e-01</td>
      <td>7.046831e-01</td>
      <td>6.769805e-01</td>
      <td>1.170944e-02</td>
      <td>1.164661e-02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.034083e+00</td>
      <td>3.212478e+00</td>
      <td>4.181148e+00</td>
      <td>3.067960e+00</td>
      <td>2.826333e+00</td>
      <td>2.687122e+00</td>
      <td>4.408512e+00</td>
      <td>4.128816e+00</td>
    </tr>
  </tbody>
</table>
</div>



## Q2. Scaling
Scaling을 통해 우리가 하고자 하는 것은 무엇인가요? 

- Scaling을 통해 각 Feature의 평균이 0이고 표준편차가 1이 되도록 합니다.

- Scaling 하는 이유는 변수가 갖는 가중치를 최대한 공정하게 맞추기 위함이라고 생각합니다. (데이터를 분석하기 전에)
- 예를들면, 어떤 두개의 data 간에 'RS' feature이 1만큼 차이난다고 했을때,
- 이 두 data 간에 'OBP'feature은 1이 차이나는 것이 아예 불가능하고, 그 이하의 수치(0~1)만큼만 차이가 날 수 있습니다. 
- 즉, 각 feature이 갖는 값의 영향력이 다르다고 할 수 있습니다.
- 모든 feature의 영향력이 같도록 하기 위해, 
- 모든 Feature들의 평균을 0으로, 표준편차가 1이 되도록 합니다.

# 5. Modeling 

## Q3. LogisticRegression() 모델을 만들어주세요. 그리고 만든 모델 인자값에 들어가는 값들의 의미를 설명해주세요.
- e.g LogisticRegression(random_state=0, solver='', multi_class='')

- **모델 명은 model로 만듭니다**
- random_state
- solver
- multi_class

- 구글링 하기 전의 저의 생각 : 
- random_state : 처음에 beta값을 난수로 생산할 때 필요한 random number generator의 seed 값
- solver : Logistic Regression model을 구현하는 방법을 정하는 인자 같습니다.
- multi_class : X값이 어떤 형태로 주어지는지에 대한 인자 같습니다.

- 구글링 해서 찾은 Ref : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- random_state : 데이터를 shuffle 하는데 필요한 generaor가 사용할 seed 값
- solver : optimization시에 사용할 알고리즘 종류 선택. 
- 가능한 값 : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} 
- liblinear'은 binary probelm에 적절, ‘lbfgs’은 multiclass problem에 적절
- multi_class : 구분해야하는 클래스의 갯수 설정 
- 가능한 값 : str, {‘ovr’, ‘multinomial’, ‘auto’}, optional (default=’ovr’)
- 'ovr' : binary problem, 'multinomial' : multiclass problem


```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr').fit(X_train, y_train) 
#playoff가 가능한 값이 두가지이므로, multi_class 를 'ovr'로 설정하고, solver를 그에 적합한 'libnear'로 설정했습니다.
model.predict(X_test)
model.score(X_test,y_test)
```




    0.8502024291497976




```python
model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
#solver 인자 비교용으로 output이 multi_class로 가정하고 실험했습니다.
model.predict(X_test)
model.score(X_test,y_test)
```




    0.8582995951417004



## Q4. data를 교차검증 해주세요.(10-fold cross_validation)

- 10-fold cross_validation을 위한 인자값을 입력해주세요.
- kfold = selection.KFold("교차검증을 위한 인자 만들기")
- 교차검증 결과를 출력하고 해석합니다.



```python
model = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr').fit(X_train, y_train) 
```


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)
```


```python
print(scores)
```

    [0.89516129 0.87096774 0.84677419 0.91935484 0.90243902 0.86178862
     0.84552846 0.86178862 0.91803279 0.82786885]



```python
scores.mean()
```




    0.8749704419307547




```python
f,ax=plt.subplots(1,1,figsize=(5,5))
sns.distplot(scores)
ax.set_xlim(0.75,1)
ax.set_xticks(np.linspace(0.75,1,5))
plt.show()
```


![png](Week2_Logistic_wk2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week2_Logistic_wk2_%EA%B8%B8%ED%83%9C%ED%98%95_126_0.png)


- 10회의 교차검증 결과가 모두 같지는 않지만, 0.82~0.93 구간에 존재합니다.

- Ref : https://datascienceschool.net/view-notebook/266d699d748847b3a3aa7b9805b846ae/

## 6.  Feature Selection
- ref: https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

- 중요 Feature를 선택하는 방법입니다.
- Kaggle 자료를 참고하였습니다.
- 내가 만든 모델에서 어떤 변수가 중요한지 한 번 살펴보세요


```python
X.head()
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
      <th>League</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>League</th>
      <th>RS</th>
      <th>RA</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
      <th>Playoffs</th>
      <th>G</th>
      <th>OOBP</th>
      <th>OSLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>734</td>
      <td>688</td>
      <td>0.328</td>
      <td>0.418</td>
      <td>0.259</td>
      <td>0</td>
      <td>162</td>
      <td>0.317</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>700</td>
      <td>600</td>
      <td>0.320</td>
      <td>0.389</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.306</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>712</td>
      <td>705</td>
      <td>0.311</td>
      <td>0.417</td>
      <td>0.247</td>
      <td>1</td>
      <td>162</td>
      <td>0.315</td>
      <td>0.403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>734</td>
      <td>806</td>
      <td>0.315</td>
      <td>0.415</td>
      <td>0.260</td>
      <td>0</td>
      <td>162</td>
      <td>0.331</td>
      <td>0.428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>613</td>
      <td>759</td>
      <td>0.302</td>
      <td>0.378</td>
      <td>0.240</td>
      <td>0</td>
      <td>162</td>
      <td>0.335</td>
      <td>0.424</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_selection import RFE

cols = ["BA", "League", "OOBP", "OSLG", "RA", "RS", "SLG"]
X = data[cols]
y = data['Playoffs']

# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

```

    Selected features: ['OSLG', 'RA', 'RS']


    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


- RFE (recursive feature elimination)는 Backward 방식중 하나로, 모든 변수를 우선 다 포함시킨 후 반복해서 학습을 진행하면서 중요도가 낮은 변수를 하나씩 제거하는 방식이다.
- 인자로 몇개의 변수를 남길 것인지 input 받습니다.

- Ref : https://wikidocs.net/16599

중요 Feature를 3개 정도 뽑아봤는데
Batting Average가 포함되지 않았다. 신기하다.

Q. How to calculate Odds ratio?

https://stackoverflow.com/questions/38646040/attributeerror-linearregression-object-has-no-attribute-coef


```python
model.fit(X, y)
model.coef_
```

    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    array([[-0.01307594,  0.00790457, -0.02705128, -0.03298225, -0.03250142,
             0.02834823, -0.00785103]])




```python
model.coef_
```




    array([[-0.01307594,  0.00790457, -0.02705128, -0.03298225, -0.03250142,
             0.02834823, -0.00785103]])



- 이 값이 odds ratio 값들입니다.


```python
X.head()
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
      <th>BA</th>
      <th>League</th>
      <th>OOBP</th>
      <th>OSLG</th>
      <th>RA</th>
      <th>RS</th>
      <th>SLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.259</td>
      <td>1</td>
      <td>0.317</td>
      <td>0.415</td>
      <td>688</td>
      <td>734</td>
      <td>0.418</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.247</td>
      <td>1</td>
      <td>0.306</td>
      <td>0.378</td>
      <td>600</td>
      <td>700</td>
      <td>0.389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.247</td>
      <td>0</td>
      <td>0.315</td>
      <td>0.403</td>
      <td>705</td>
      <td>712</td>
      <td>0.417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.260</td>
      <td>0</td>
      <td>0.331</td>
      <td>0.428</td>
      <td>806</td>
      <td>734</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.240</td>
      <td>1</td>
      <td>0.335</td>
      <td>0.424</td>
      <td>759</td>
      <td>613</td>
      <td>0.378</td>
    </tr>
  </tbody>
</table>
</div>





## 축하드립니다. 여러분은 이제 로지스틱 모델을 구현하실 수 있게 되었습니다!
