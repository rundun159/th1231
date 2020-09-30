```python
import sys
import numpy as np
from House import Apartment, Vile
```


```python
Apartment()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-117-d3aeb1a562fb> in <module>
    ----> 1 Apartment()
    

    TypeError: __init__() missing 3 required positional arguments: 'dong_ho', 'size', and 'room'



```python
f = open("Apartment.txt", 'r')
N=int(f.readline())
```


```python
apartments=[]
```


```python
params = [None]*4
listsArr=[]
```


```python
for i in range(N):
    lists=f.readline().split(' ')
    listsArr.append(lists)
f.close()

```


```python
listsArr[0][0]
```




    '207-1105'




```python
Apartment('asd',1,2)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-106-393d904c551d> in <module>
    ----> 1 Apartment('asd',1,2)
    

    ~\Google 드라이브\Study\BIgData\Week5\5주차_클래스\클래스과제_배포\House.py in __init__(self, dong_ho, size, room)
         15     def __str__(self):
         16         return str(self.addr.split('-')[0])+'동 '+str(self.addr.split('-')[1])+'호의 '+str(self.room)+'개의 방이 있는 '+str(self.size)+'평의 아파트입니다.'
    ---> 17 
         18 class Vile(House):
         19     yard=False


    AttributeError: 'Apartment' object has no attribute 'split'



```python
print(range(5))
```

    range(0, 5)



```python
for i in range(5):
    print(i)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-109-d8c179fd2957> in <module>
    ----> 1 for i in range(5)+1:
          2     print(i)


    TypeError: unsupported operand type(s) for +: 'range' and 'int'



```python
dir()
```
