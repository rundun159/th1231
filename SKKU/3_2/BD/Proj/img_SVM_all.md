```python
from PIL import Image
import numpy as np
import pickle

```


```python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
```


```python
PCA_IMG=20000
```


```python
img_idx = format(1,'06d')
im = np.array(Image.open('./img_files/img_align_celeba/'+img_idx+'.jpg'))
im = rgb2gray(im)
train_img = im.reshape(1,-1)
for num,idx in enumerate(range(2,PCA_IMG)):
    if(num%500==0):
        print("%d %d " %(num, idx))
    img_idx = format(idx,'06d')
    im = np.array(Image.open('./img_files/img_align_celeba/'+img_idx+'.jpg'))
    im_gray = rgb2gray(im)
    del im
    train_img=np.concatenate((train_img,im_gray.reshape(1,-1)),axis=0)
```

    0 2 
    500 502 
    1000 1002 
    1500 1502 
    2000 2002 
    2500 2502 
    3000 3002 
    3500 3502 
    4000 4002 
    4500 4502 
    5000 5002 
    5500 5502 
    6000 6002 



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-1e9472608a50> in <module>
          7         print("%d %d " %(num, idx))
          8     img_idx = format(idx,'06d')
    ----> 9     im = np.array(Image.open('./img_files/img_align_celeba/'+img_idx+'.jpg'))
         10     im_gray = rgb2gray(im)
         11     del im


    ~\Anaconda3\lib\site-packages\PIL\Image.py in open(fp, mode)
       2768 
       2769     if filename:
    -> 2770         fp = builtins.open(filename, "rb")
       2771         exclusive_fp = True
       2772 


    KeyboardInterrupt: 



```python
with open('./pickles/img_total', 'wb') as f:
    pickle.dump(train_img,f)
    
```


```python
with open('./pickles/train_img', 'wb') as f:
    pickle.dump(train_img,f)
    
with open('./pickles/test_img', 'wb') as f:
    pickle.dump(test_img,f)
```


```python
train_img.shape
```




    (1400, 38804)




```python
test_img.shape
```




    (600, 38804)




```python

```
