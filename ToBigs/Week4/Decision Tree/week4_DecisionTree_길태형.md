# DT Assignment1

# Data Loading


```python
import pandas as pd 
import numpy as np
import math
```


```python
pd_data = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
pd_data.drop("RID",axis=1, inplace = True) #RID는 그냥 순서라서 삭제
pd_data
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
      <th>income</th>
      <th>student</th>
      <th>credit_rating</th>
      <th>class_buys_computer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>youth</td>
      <td>high</td>
      <td>no</td>
      <td>fair</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>youth</td>
      <td>high</td>
      <td>no</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>middle_aged</td>
      <td>high</td>
      <td>no</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>senior</td>
      <td>medium</td>
      <td>no</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>senior</td>
      <td>low</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>senior</td>
      <td>low</td>
      <td>yes</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>middle_aged</td>
      <td>low</td>
      <td>yes</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>youth</td>
      <td>medium</td>
      <td>no</td>
      <td>fair</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>youth</td>
      <td>low</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>senior</td>
      <td>medium</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>youth</td>
      <td>medium</td>
      <td>yes</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>middle_aged</td>
      <td>medium</td>
      <td>no</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>middle_aged</td>
      <td>high</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>senior</td>
      <td>medium</td>
      <td>no</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



# 1. Gini 계수를 구하는 함수 만들기

- Input: df(데이터), label(타겟변수명)
- 해당 결과는 아래와 같이 나와야 합니다.


```python
def get_gini(df, label):
    dummy_var = pd.get_dummies(df[label])    #Feature의 각 클래스에 대해서 one-hot Encoding해줍니다.
    gini=1                                    #gini = 1 -((각 클래스에 속하는 확률)^2의 총 클래스 합)
    sizeofData = len(df)                      #데이터의 전체 갯수. P값의 분모로 대입됨.
    for i in dummy_var.columns:               #각 클래스의 이름을 i에 저장하고 반복문을 수행하게 됩니다. 
        numofOneClass = dummy_var[i].sum()       #각 클래스의 속하는 데이터의 갯수. P의 분자로 대입됨.
        gini -= (numofOneClass/sizeofData) *(numofOneClass/sizeofData)  #P의 제곱을 원래의 gini값에서 빼줍니다.
    return gini    
```


```python
get_gini(pd_data,'class_buys_computer')
```




    0.4591836734693877



# 2. Feature의 Class를 이진 분류로 만들기
 ## ex) {A,B,C} -> ({A}, {B,C}), ({B}, {A,C}), ({C}, {A,B})

- Input: df(데이터), attribute(Gini index를 구하고자 하는 변수명)
- 해당 결과는 아래와 같이 나와야 합니다.


```python
def makeSplit(SIdx, num, container, preV):    
    #<인자들의 type>
    #SIdx,num : int type, container : numpy를 담는 numpy 배열, preV: bool type numpy 배열
    #<인자들 설명>
    #preV는 그 전까지 선택한 원소들의 정보를 담는 numpy
    #SIdx(startIndex)를 포함해서, 그 뒤로 num개의 원소를 선택해서, 그 정보를 preV에 추가해서 container에 넣어줌.
    #포함하는 class를 bool encoding으로 반환
    #<함수 설명>
    #재귀호출 형식으로 0,1 encoding으로 포함시킬 원소들을 저장함.
    if(SIdx>=len(preV)):          #SIdx가 out of Bound 일 경우.
        return
    if((len(preV)-SIdx)<num):    #SIdx이후로 num개의 원소를 선택할 수 없는 경우.
        return
    newV=np.array(preV)
    newV[SIdx]=1                #SIdx에 해당하는 원소를 포함시킴.
    if (num==1):                 #다 포함시킨 경우, 해당 경우를 container에 포함시키고 반환함.
        container.append(newV)
        del newV
        return
    for i in range(SIdx+1,len(preV)):              #현재 Index의 다음 Index부터 시작해서, 끝까지의 index를 SIdx로 지정하고 재귀호출함.
            makeSplit(i,num-1,container,newV)
    del newV
```


```python
def get_classes(classes,boolArr,container): 
    #<인자들의 type>
    #clases : string을 담는 numpy. boolArr : 0,1을 저장한 numpy. container : 클래스 split을 저장하는 numpy.
    #<인자들 설명>
    #clases는 해당 attribute의 class들
    #boolArr은 선택할 attribute의 class들. 0,1로 encoding
    #container은 split한 클래스들을 저장하는 numpy
    #<함수 설명>
    #0,1형식으로 주어진 선택할 class들을 string 형태로 저장함.
    #new2Split[0]에는 1로 coding된 원소들, new2Split[1]에는 0으로 coding된 원소들을 저장함.

    #boolArr의 길이가 짝수이고, boolArr의 원소 중 값이 1인 원소가 절반만큼 있는 경우에는,
    #예를 들어 [1,1,0,0] 과 [0,0,1,1]을 이용해 만드는 classes배열은 중복되므로
    #한번만 만들어야함.
    if (boolArr.sum()*2) == len(boolArr):
        new2Splits=[[]]
        for i in range(0,len(boolArr)):                   #인자로 받은 boolArr을 하나씩 참조함
            if(boolArr[i]):                               #1로 coding되어 있는 원소는 new2Split[0]에 저장.
                new2Splits[0].append(classes[i])
        container.append(new2Splits[0])   
    else:
        new2Splits = [[],[]]        
        for i in range(0,len(boolArr)):                   #인자로 받은 boolArr을 하나씩 참조함
            if(boolArr[i]):                               #1로 coding되어 있는 원소는 new2Split[0]에 저장.
                new2Splits[0].append(classes[i])
            else:                                        #0으로 coding되어 있는 원소는 new2Split[1]에 저장. 
                new2Splits[1].append(classes[i])
        container.append(new2Splits[0])   
        container.append(new2Splits[1])
    del new2Splits
```


```python
def get_boolArrs(df,attribute): 
    #인자 설명은 주어진 함수와 동일하므로 생략합니다.
    #<함수 설명>
    #가능한 split을 0,1 encoding으로 반환합니다.
    container = []
    numofClasses = len(df[attribute].unique())
    for i in range(0,numofClasses):
        for j in range(1,int(numofClasses/2+1)):
            makeSplit(i,j,container,np.zeros(numofClasses))
    return container
```


```python
def get_binary_split(df, attribute):
    result=[]
    boolSet=get_boolArrs(df,attribute)                 #split을 0,1 encoding으로 저장함.
    for i in boolSet:                  #coding된 원소들을 하나씩 원래의 class명으로 coding합니다.
        get_classes(df[attribute].unique(),i,result)
    return result
```


```python
get_binary_split(pd_data,'age')
```




    [['youth'],
     ['middle_aged', 'senior'],
     ['middle_aged'],
     ['youth', 'senior'],
     ['senior'],
     ['youth', 'middle_aged']]



# 3. 다음은 모든 이진분류의 경우의 Gini index를 구하는 함수 만들기
- 위에서 완성한 두 함수를 사용하여 만들어주세요!
- 해당 결과는 아래와 같이 나와야 합니다.


```python
def getStr(strArr):
    result=strArr[0]
    for i in range(1,len(strArr)):
        result+='_'+strArr[i]
    return result
```


```python
def get_attribute_gini_index(df, attribute, label):
    #result를 dict type으로 초기화.
    result={}
    sizeofData = len(df)
    classes = get_binary_split(df,attribute)
    
    for i in classes:
        gini = 0 
        sizeofAClass = len(df.loc[df[attribute].isin(i)])         #해당 attribute가 현재 탐색중인 class값을 가지는 data 갯수
        sizeofNotAClass = sizeofData - sizeofAClass               #현재 탐색중인 class값을 갖지 않는 data 갯수
        gini = get_gini(df.loc[df[attribute].isin(i)],label)*sizeofAClass/sizeofData          #gini계수에 해당 class 값을 갖는 data의 비율을 곱해줌.
        gini += get_gini(df.loc[~(df[attribute].isin(i))],label)*sizeofNotAClass/sizeofData   #gini계수에 해당 class 값을 갖지 않는 data의 비율을 곱해줌.
        result[getStr(i)]=gini          #dict형태로 저장함
    return result
```


```python
get_attribute_gini_index(pd_data, "age", "class_buys_computer")
```




    {'youth': 0.3936507936507936,
     'middle_aged_senior': 0.3936507936507936,
     'middle_aged': 0.35714285714285715,
     'youth_senior': 0.35714285714285715,
     'senior': 0.45714285714285713,
     'youth_middle_aged': 0.45714285714285713}



여기서 가장 작은 Gini index값을 가지는 class를 기준으로 split해야겠죠?


```python
min(get_attribute_gini_index(pd_data, "age", "class_buys_computer").items())
```




    ('middle_aged', 0.35714285714285715)



# 다음의 문제를 위에서 작성한 함수를 통해 구한 값으로 보여주세요!
## 문제1) 변수 ‘income’의 이진분류 결과를 보여주세요.

## 문제2) 분류를 하는 데 가장 중요한 변수를 선정하고, 해당 변수의 Gini index를 제시해주세요.

## 문제3) 문제 2에서 제시한 feature로 DataFrame을 split한 후 나눠진 2개의 DataFrame에서 각각   다음으로 중요한 변수를 선정하고 해당 변수의 Gini index를 제시해주세요.


```python
##문제1 답안
get_binary_split(pd_data,'income')
```




    [['high'],
     ['medium', 'low'],
     ['medium'],
     ['high', 'low'],
     ['low'],
     ['high', 'medium']]




```python
##문제2 답안
def retMinGini(df,target):
    #<함수 설명>
    #주어진 target변수에 대해서, 
    #Gini index를 ret[0]에 저장하고,
    #변수 명을 ret[1]에 저장합니다.
    ret = [('',999),'']        #최소 Gini index를 저장할 변수입니다. Gini 계수를 큰값으로 초기화합니다.
    col = df.columns           #주어진 data가 갖는 변수들의 list입니다. 
    for i in col:             
        if i!=target:
            gini = [min(get_attribute_gini_index(df, i, target).items()),i]                #각 변수의 Gini index를 계산합니다.
            if gini[0][1]<ret[0][1]:                      #현재 최소 GIni index와 비교하고
                ret=gini                                  #더 작다면 갱신합니다.
    return ret    
```


```python
minG=retMinGini(pd_data,'class_buys_computer')
```


```python
#분류 하는데 가장 중요한 변수
minG[1]
```




    'age'




```python
#해당 변수의 Gini index
minG[0]
```




    ('middle_aged', 0.35714285714285715)




```python
##문제3 답안
def retMinGiniExpand(df,target,filtered):
    #filtered: 이미 split으로 사용한 변수 list
    #<함수 설명>
    #첫번째 split이후에 사용될 함수
    #filtered된 변수 말고, 다른 변수들 중에서 최소 Gini index를 갖는 변수를 찾는 함수.
    
    #data에서 filtered에 포함된 변수들을 제거할수도 있겠지만,
    #최대한 data를 조작하지 않으면서 진행하고 싶었습니다.

    ret = [('',999),'']        #최소 Gini index를 저장할 변수입니다. Gini 계수를 큰값으로 초기화합니다.
    col = df.columns           #주어진 data가 갖는 변수들의 list입니다. 
    for i in col:             
        isFiltered = False
        for j in filtered:     #이미 split으로 사용한 변수일 경우, 생략합니다.
            if i==j:
                isFiltered=True
                break
        if(isFiltered):
                continue
        if (i!=target):
            gini = [min(get_attribute_gini_index(df, i, target).items()),i]                #각 변수의 Gini index를 계산합니다.
            if gini[0][1]<ret[0][1]:                      #현재 최소 GIni index와 비교하고
                ret=gini                                  #더 작다면 갱신합니다.
    return ret        
```


```python
#df1, df2는 split한 후 나눠진 두개의 Data Frame
df1 = pd.DataFrame()
df2 = pd.DataFrame()
```


```python
filtered=[minG[1]]
```


```python
df1=pd_data.loc[pd_data[minG[1]]==minG[0][0]]
```


```python
df1
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
      <th>income</th>
      <th>student</th>
      <th>credit_rating</th>
      <th>class_buys_computer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>middle_aged</td>
      <td>high</td>
      <td>no</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>middle_aged</td>
      <td>low</td>
      <td>yes</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>middle_aged</td>
      <td>medium</td>
      <td>no</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>middle_aged</td>
      <td>high</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
minG2=retMinGiniExpand(df1,'class_buys_computer',filtered)
```


```python
#Split된 첫번째 DataFrame에서 가장 중요한 변수
minG2[1]
```




    'income'




```python
#그 변수의 GIni Index
minG2[0]
```




    ('high', 0.0)



- df1을 'income'변수가 'high'인지를 기준으로 split을 한다면, 
- 두개의 split의 target 값이 모두 yes이고, 
- 두개의 split의 크기가 같으므로
- Gini값은 0이 됩니다.


```python
df2=pd_data.loc[~(pd_data[minG[1]]==minG[0][0])]
```


```python
df2
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
      <th>income</th>
      <th>student</th>
      <th>credit_rating</th>
      <th>class_buys_computer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>youth</td>
      <td>high</td>
      <td>no</td>
      <td>fair</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>youth</td>
      <td>high</td>
      <td>no</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>senior</td>
      <td>medium</td>
      <td>no</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>senior</td>
      <td>low</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>senior</td>
      <td>low</td>
      <td>yes</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>youth</td>
      <td>medium</td>
      <td>no</td>
      <td>fair</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>youth</td>
      <td>low</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>senior</td>
      <td>medium</td>
      <td>yes</td>
      <td>fair</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>youth</td>
      <td>medium</td>
      <td>yes</td>
      <td>excellent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>senior</td>
      <td>medium</td>
      <td>no</td>
      <td>excellent</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
minG2=retMinGiniExpand(df2,'class_buys_computer',filtered)
```


```python
#Split된 두번째 DataFrame에서 가장 중요한 변수
minG2[1]
```




    'student'




```python
#그 변수의 GIni Index
minG2[0]
```




    ('no', 0.31999999999999984)


