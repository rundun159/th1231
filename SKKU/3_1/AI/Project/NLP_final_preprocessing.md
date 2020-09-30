```python
import json # import json module
import numpy as np
import csv
import pickle
import math
import codecs
import NLP_functions as NLP
```


```python
%load_ext autoreload
%autoreload 2
```

# Train Data Preprocessing

# 각 Docment안에 있는 단어들의 tf 값 구하기. Dictionary type으로 저장

### Tr file의 json data를 가져옵니다.


```python
with codecs.open('Tr.json', 'r', encoding='cp949', errors='replace') as train_file:
    train_data = json.load(train_file)
```

### 불용어를 가져옵니다.
- Korean Stopwords Ref : https://www.ranks.nl/stopwords/korean
- csv파일 형태로 stopWords를 저장했고,
- docment에서 단어를 추출할 때, stopwords에 포함 되는 단어는 제거하는 방식으로 사용했습니다.


```python
#불용어 가져오기.
stopwords_dict=NLP.stopWords()   
```

### 불용어(stopWords)에 있는 단어들을 제외하면서, json data에 있는 문서들의 tf를 계산합니다.
- 처음에는 형태소가 N과 V로 시작하는 단어들을 corpus에 추가했으나,
- V로 시작하는 단어들을 살펴보니, 문서의 종류와 관련이 없는 단어들이 많았고,
- 학습을 용이하게 하고, 연산을 빠르게 하기 위해 Input Data의 Feature의 개수를 줄이고 싶었습니다. 
- 그래서, Test Data에서 형태소가 N으로 시작하는 단어들만 corpus에 추가했습니다. 


```python
train_dict_list=NLP.ret_dict_list(train_data,stopwords_dict) 
```

### train data의 정답 data를 가지고, one-hot encoding을 진행합니다.  


```python
train_labels,empty_labels_list=NLP.ret_labels(train_data)
```

###  필요가 없어진 json data는 삭제합니다.


```python
del train_data
```

### 현재까지 process한 data를 다른 파일에서도 사용 할 수 있도록 pickle로 저장합니다.


```python
with open('train_dict_list.pickle', 'wb') as f:
    pickle.dump(train_dict_list, f, pickle.HIGHEST_PROTOCOL)
```

# Corpus를 만들고, 각 단어의 df값을 구합니다. 
# Dictionary type으로 저장. 각 element는 (key,value)

- 이미 계산한 각 문서들의 tf data를 이용하여, corpus를 만들고, 각 단어의 df를 저장합니다.
- train_corpus_dict 변수에 저장합니다.


```python
train_corpus_dict = NLP.ret_corpus_dict(train_dict_list)
with open('train_corpus_dict.pickle', 'wb') as f:
        pickle.dump(train_corpus_dict, f, pickle.HIGHEST_PROTOCOL)
```

# 만든 corpus df Dictionary를 바탕으로 Numpy type으로 바꿉니다. 
- 각 element는 (key,value). value는 각 단어의 df
- train_corpus_np 변수에 저장합니다.


```python
train_corpus_np=NLP.ret_corpus_np(train_corpus_dict,len(train_dict_list))
with open('train_corpus_np.pickle', 'wb') as f:
        pickle.dump(train_corpus_np, f, pickle.HIGHEST_PROTOCOL)
```

# train data의 각 document의 tf-idf값을 구합니다.
- numpy 형태로 저장. key값 없이 tf-idf만 저장. 
- doc_tf_idf_list 에 각 문서들의 tf-idf 벡터가 저장됩니다.
- empty_data_list에 단어가 아예 없는 문서들의 index를 저장합니다. (추후에 제거 예정)


```python
doc_tf_idf_list, empty_data_list = NLP.ret_doc_tf_idf(train_corpus_dict, train_corpus_np, train_dict_list)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3480/3480 [03:49<00:00, 15.13it/s]


# list의 형태로 저장되어 있는 data를 numpy type으로 변환합니다. 


```python
doc_tf_idf_np=np.asarray(doc_tf_idf_list)
```

# train_data에서 비어있는 document를 제거합니다
- 위에서 구했던 empty_list를 이용해서 input data와 label data 모두에서 제거합니다.


```python
empty_list= empty_labels_list + empty_data_list
empty_list=list(set(empty_list))
```


```python
empty_list.sort()
```

- 어떤 문서들이 비어있는지 확인해봤습니다.


```python
print(empty_list)
```

    [2093, 2158, 2244]


- 정말로 비어있는 문서가 맞는지, 해당 문서의 tf - idf값을 확인해봤습니다. 


```python
doc_tf_idf_np[empty_list]
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])



- 비어있는 문서를 제거하기 전의 train_data의 형태입니다.


```python
doc_tf_idf_np.shape
```




    (3480, 11789)




```python
train_labels.shape
```




    (3480, 5)



### 비어 있는 문서를 제거합니다.


```python
doc_tf_idf_np=np.delete(doc_tf_idf_np,empty_list,axis=0)
train_labels=np.delete(train_labels,empty_list,axis=0)
```

# 제거하고 난 후, 최종적인 train data의 형태입니다.


```python
doc_tf_idf_np.shape
```




    (3477, 11789)




```python
train_labels.shape
```




    (3477, 5)




```python
with open('doc_tf_idf_np.pickle', 'wb') as f:
    pickle.dump(doc_tf_idf_np, f, pickle.HIGHEST_PROTOCOL)
with open('train_labels.pickle', 'wb') as f:
    pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)
with open('train_data.pickle', 'wb') as f:
    pickle.dump(doc_tf_idf_np, f, pickle.HIGHEST_PROTOCOL)
```

# TestData Preprocessing

- TrainData Preprocessing에서 진행한 과정과 유사합니다.
- 불용어를 제외한 단어들을 저장하고,
- 각 문서의 tf 값을 dictionary type으로 저장하고,
- 정답 data를 one-hot encoding합니다.


```python
with codecs.open('Te.json', 'r', encoding='cp949', errors='replace') as test_file:
    test_data = json.load(test_file)
stopwords_dict=NLP.stopWords()
test_dict_list=NLP.ret_dict_list(test_data,stopwords_dict)
test_labels,_=NLP.ret_labels(test_data)
del test_data
with open('test_dict_list.pickle', 'wb') as f:
    pickle.dump(test_dict_list, f, pickle.HIGHEST_PROTOCOL)
```


```python
len(test_dict_list)
```




    1497




```python
len(test_labels)
```




    1497



# test data의 각 document의 tf-idf값을 구합니다.
- numpy 형태로 저장. key값 없이 tf-idf만 저장. 
- test_data_list 에 각 문서들의 tf-idf 벡터가 저장됩니다.


```python
test_data_list = NLP.ret_doc_tf_idf_test(train_corpus_dict, train_corpus_np, test_dict_list)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1497/1497 [01:29<00:00, 16.74it/s]



```python
test_data=np.asarray(test_data_list)
```


```python
test_data.shape
```




    (1497, 11789)




```python
test_data[0].shape
```




    (11789,)



####  normalize가 잘 되어있는지 확인 해봤습니다.


```python
np.sum(test_data[0])
```




    0.9999999999999998



## 다른 파일에서 데이터를 사용할 수 있도록 pickle로 저장했습니다.


```python
with open('test_data.pickle', 'wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
```


```python
with open('test_labels.pickle', 'wb') as f:
    pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)
```

## MLP구현 파트로 이어집니다.
