```python
import numpy as np
from numpy.linalg import inv 

def estimate_beta(x, y):
    """구현해야 하는 부분"""
    x_inv = np.linalg.inv(x)
    x_square = np.dot(x_inv, x)
    beta_hat = np.dot(np.dot(np.linalg.inv(x_square), x_inv),y)
    # (X,.T*X)^-1 * X.t * Y 를 구현했습니다.
    return beta_hat
```

https://sdong3161.tistory.com/5 을 참고했습니다.


```python
#x행렬에 intercept를 추가해야함
x = np.array([ [1, 0, 1],
               [1, 2, 3],
               [1, 3, 8]]) #3x3
y = np.transpose(np.array([1, 3, 7])) #3x1
```


```python
estimate_beta(x,y)
```




    array([0.25, 0.25, 0.75])




```python
#sklearn은 디폴트로 intercept가 fit되므로, 데이터에 intercept가 포함된 경우는 fit_intercept=False로 해야함
```


```python
from sklearn.linear_model import LinearRegression
```


```python
reg = LinearRegression(fit_intercept=False).fit(x, y)
```


```python
reg.coef_
```




    array([0.25, 0.25, 0.75])


