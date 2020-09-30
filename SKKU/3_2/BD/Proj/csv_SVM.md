```python
import numpy as np
import pandas as pd
import pickle
train_frac = 0.8
test_frac = 0.3
TH_DATA_LEN=2000
data=pd.read_csv('./csv_files/list_attr_celeba.csv')
```


```python
data_len = len(data)
```


```python
train_len = int(TH_DATA_LEN *train_frac)

idx=np.asarray(range(data_len))

np.random.shuffle(idx)

train_idx = idx[:train_len]

test_idx = idx[train_len:TH_DATA_LEN]
```


```python
train_idx.shape
```




    (1600,)




```python
test_idx.shape
```




    (400,)




```python
data_label = data['Male']

data_x = data[data.columns.difference(['Male','image_id'])]

train_x = data_x.iloc[train_idx]

test_x=data_x.iloc[test_idx]

train_y = data_label[train_idx]

test_y = data_label[test_idx]

train_x=np.asarray(train_x)

test_x = np.asarray(test_x)

train_y=np.asarray(train_y)

test_y = np.asarray(test_y)

```


```python
train_x.shape
```




    (1600, 39)




```python
test_x.shape
```




    (400, 39)




```python
train_y.shape
```




    (1600,)




```python
test_y.shape
```




    (400,)




```python
with open('./pickles/train_csv_x','wb') as f:
    pickle.dump(train_x,f)

with open('./pickles/test_csv_x','wb') as f:
    pickle.dump(test_x,f)

with open('./pickles/train_csv_y','wb') as f:
    pickle.dump(train_y,f)

with open('./pickles/test_csv_y','wb') as f:
    pickle.dump(test_y,f)
    
with open('./pickles/train_idx', 'wb') as f:
    pickle.dump(train_idx,f)
    
with open('./pickles/test_idx', 'wb') as f:
    pickle.dump(test_idx,f)
```


```python
train_x.shape
```




    (1600, 39)




```python
test_x.shape
```




    (400, 39)




```python
train_idx.shape
```




    (1600,)




```python
test_idx.shape
```




    (400,)




```python

```


```python

```
