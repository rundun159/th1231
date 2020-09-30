```python
import cv2
```


```python
from PIL import Image
import numpy as np
import pickle
```


```python
with open('./pickles/train_idx', 'rb') as f:
    train_idx = pickle.load(f)
with open('./pickles/test_idx', 'rb') as f:
    test_idx = pickle.load(f)
```


```python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
```


```python
img_idx = format(train_idx[10],'06d')
```


```python
img_idx = format(train_idx[14],'06d')
img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.Canny(img, 50, 200)
    
```


```python
cv2.imshow('edge detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
img_idx = format(train_idx[0],'06d')
img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.Canny(img, 50, 200)
```


```python
img
```




    array([[  0,   0,   0, ...,   0,   0,   0],
           [  0,   0,   0, ...,   0,   0,   0],
           [  0,   0,   0, ...,   0,   0,   0],
           ...,
           [  0,   0,   0, ...,   0,   0,   0],
           [  0, 255, 255, ...,   0,   0,   0],
           [255, 255,   0, ...,   0,   0,   0]], dtype=uint8)




```python
cv2.imshow('messy  edge',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```


```python
cv2.imshow(img)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-85-3594ed6a38c9> in <module>
    ----> 1 cv2.imshow(img)
    

    TypeError: imshow() missing required argument 'mat' (pos 2)



```python
PART = int(len(train_idx)/10)
PART
```




    2000




```python
PART = int(len(train_idx)/10)
for i in range(10):
    img_idx = format(train_idx[i*PART],'06d')
    img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
    img=cv2.Canny(img, 50, 200)
    train_img = np.asarray(img.reshape(1,-1).tolist())
    for num,idx in enumerate(train_idx[i*PART+1:(i+1)*PART]):
        if(num%500==0):
            print("%d %d " %(num, idx))
        img_idx = format(train_idx[0],'06d')
        img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
        img=cv2.Canny(img, 50, 200)
        train_img=np.concatenate((train_img,img.reshape(1,-1).tolist()),axis=0)
        del img
    with open('./pickles/train_img_set'+str(i), 'wb') as f:
        pickle.dump(train_img,f)
    del train_img
```

    0 193160 
    500 73047 
    1000 43291 
    1500 140634 
    0 68268 
    500 65048 
    1000 8171 
    1500 20937 
    0 7434 
    500 60155 
    1000 113589 
    1500 28319 
    0 151428 
    500 163101 
    1000 36441 
    1500 171072 
    0 95683 
    500 40263 
    1000 36739 
    1500 34759 
    0 35743 
    500 106978 
    1000 169146 
    1500 34460 
    0 17497 
    500 195815 
    1000 88265 
    1500 190621 
    0 188976 
    500 148971 
    1000 31240 
    1500 75696 
    0 176558 
    500 193529 
    1000 73025 
    1500 125637 
    0 23707 
    500 173596 
    1000 1860 
    1500 94523 



```python
img_idx = format(train_idx[0],'06d')
img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.Canny(img, 50, 200)
train_img = np.asarray(img.reshape(1,-1).tolist())
for num,idx in enumerate(train_idx[1:]):
    if(num%500==0):
        print("%d %d " %(num, idx))
    img_idx = format(train_idx[0],'06d')
    img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
    img=cv2.Canny(img, 50, 200)
    train_img=np.concatenate((train_img,img.reshape(1,-1).tolist()),axis=0)
    del img

```

    0 103495 
    500 190684 
    1000 81973 
    1500 48299 



```python
img_idx = format(test_idx[0],'06d')
img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.Canny(img, 50, 200)
test_img = np.asarray(img.reshape(1,-1).tolist())
for num,idx in enumerate(test_idx[1:]):
    if(num%500==0):
        print("%d %d " %(num, idx))
    img_idx = format(test_idx[0],'06d')
    img = cv2.imread('./img_files/img_align_celeba/'+img_idx+'.jpg',cv2.IMREAD_GRAYSCALE)
    img=cv2.Canny(img, 50, 200)
    test_img=np.concatenate((test_img,img.reshape(1,-1).tolist()),axis=0)
    del img

```

    0 184564 



```python
cv2.imshow('messy  edge',train_img[0])

cv2.waitKey(0)
cv2.destroyAllWindows()
```


    ---------------------------------------------------------------------------

    error                                     Traceback (most recent call last)

    <ipython-input-90-66abb8781a89> in <module>
    ----> 1 cv2.imshow('messy  edge',train_img[0])
          2 
          3 cv2.waitKey(0)
          4 cv2.destroyAllWindows()


    error: OpenCV(4.2.0) C:/projects/opencv-python/opencv/modules/highgui/src/precomp.hpp:137: error: (-215:Assertion failed) src_depth != CV_16F && src_depth != CV_32S in function 'convertToShow'




```python
train_img=train_img/255
```


```python
test_img=test_img/255
```


```python
np.unique(test_img)
```




    array([0., 1.])




```python
train_img.shape
```




    (1600, 38804)




```python
test_img.shape
```




    (400, 38804)




```python
with open('./pickles/train_img', 'wb') as f:
    pickle.dump(train_img,f)
    
with open('./pickles/test_img', 'wb') as f:
    pickle.dump(test_img,f)
```


```python
train_img.shape
```




    (1600, 38804)




```python
test_img.shape
```




    (400, 38804)




```python

```
