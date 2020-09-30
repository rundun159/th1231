```python
import numpy as np
import pandas as pd
import pickle
```


```python
with open('./pickles/pred_result', 'rb') as f:
    pred = pickle.load(f)
with open('./pickles/test_csv_y', 'rb') as f:
    y = pickle.load(f)

    
```


```python
pred.shape
```




    (600,)




```python
y.shape
```




    (600,)




```python
np.sum(pred!=y)
```




    55




```python

```
