```python
import numpy as np
import pickle
```


```python
with open('./pickles/csv_Ham_pred_result', 'rb') as f:
    pred = pickle.load(f)

```


```python
with open('./pickles/test_csv_y', 'rb') as f:
    test_y = pickle.load(f)

```


```python
real_y_true=test_y==1

real_y_false=test_y!=1

pred_y_true=pred==1

pred_y_false=pred!=1

tp=np.sum(pred[real_y_true]==1)
print(tp)

fp=np.sum(test_y[pred_y_true]!=1)
print(fp)

fn=np.sum(test_y[pred_y_false]==1)
print(fn)

tn=np.sum(test_y[pred_y_false]!=1)
print(tn)
```


```python
print(tp+fp+fn+tn)
```

    600



```python
pred[pred==-1
```




    array([ True, False,  True,  True,  True, False, False,  True,  True,
           False,  True,  True, False,  True,  True,  True, False, False,
            True, False, False, False,  True,  True, False, False,  True,
            True, False,  True, False, False,  True,  True,  True,  True,
            True,  True, False, False, False, False,  True, False, False,
            True, False,  True, False,  True,  True, False,  True, False,
            True,  True, False, False,  True,  True, False,  True,  True,
            True, False, False,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True, False,  True,  True,  True, False,
            True, False,  True, False,  True,  True, False,  True,  True,
            True, False,  True, False,  True,  True,  True,  True, False,
           False,  True,  True,  True, False, False,  True,  True,  True,
           False,  True, False, False, False, False, False, False,  True,
            True,  True, False, False,  True, False,  True,  True,  True,
            True, False,  True,  True, False, False,  True, False, False,
           False,  True, False,  True, False,  True, False, False,  True,
           False, False, False, False,  True,  True,  True,  True,  True,
           False,  True, False, False, False, False,  True,  True,  True,
           False, False,  True,  True, False, False, False,  True, False,
           False,  True,  True, False,  True,  True,  True,  True, False,
           False, False, False,  True,  True,  True,  True,  True, False,
           False, False, False, False, False,  True, False, False,  True,
            True, False,  True,  True, False,  True, False,  True,  True,
            True, False,  True, False,  True, False, False, False, False,
            True,  True, False, False,  True,  True, False, False,  True,
           False, False, False, False, False,  True, False, False,  True,
           False,  True, False,  True, False, False,  True,  True, False,
            True,  True,  True,  True, False, False,  True, False, False,
           False,  True, False, False,  True,  True,  True,  True, False,
           False,  True,  True,  True, False,  True,  True, False,  True,
            True,  True, False, False, False,  True, False,  True,  True,
            True,  True, False, False,  True,  True,  True, False,  True,
           False,  True, False, False,  True,  True, False, False,  True,
           False, False,  True,  True, False, False,  True,  True,  True,
           False,  True, False,  True, False,  True,  True,  True,  True,
           False, False,  True,  True, False,  True, False, False,  True,
           False,  True, False, False, False,  True,  True, False,  True,
            True, False,  True,  True, False,  True,  True,  True, False,
           False, False, False,  True,  True,  True, False,  True,  True,
            True, False,  True, False, False,  True, False,  True,  True,
            True, False,  True, False, False, False,  True,  True,  True,
           False,  True, False, False,  True, False, False, False, False,
            True,  True,  True, False, False, False, False, False,  True,
           False, False,  True,  True, False,  True, False, False, False,
            True, False,  True, False, False, False,  True,  True, False,
            True,  True,  True,  True,  True, False, False, False,  True,
            True, False,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True, False,  True, False, False,  True,
            True,  True,  True, False, False,  True, False, False, False,
           False,  True,  True,  True,  True, False,  True, False, False,
            True, False,  True,  True, False, False, False, False, False,
           False, False,  True,  True,  True,  True, False, False, False,
           False,  True,  True,  True,  True,  True,  True,  True, False,
            True, False,  True,  True, False,  True,  True,  True, False,
            True,  True,  True,  True, False,  True, False, False,  True,
            True,  True, False, False, False,  True, False,  True,  True,
            True,  True,  True, False, False,  True,  True, False,  True,
            True,  True,  True,  True,  True, False,  True,  True, False,
            True,  True,  True,  True, False,  True,  True, False,  True,
           False, False,  True,  True,  True, False,  True,  True, False,
            True, False,  True, False,  True, False,  True,  True, False,
            True,  True,  True, False,  True, False,  True,  True,  True,
            True,  True,  True, False,  True, False,  True,  True, False,
           False,  True,  True,  True, False, False,  True,  True, False,
           False,  True, False, False, False,  True,  True, False,  True,
            True,  True,  True,  True, False,  True, False,  True,  True,
            True,  True,  True, False, False, False])




```python

```
