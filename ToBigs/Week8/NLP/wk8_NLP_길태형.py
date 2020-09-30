#!/usr/bin/env python
# coding: utf-8

# # 과제 요약
# #### 목적 : https://cs.skku.edu/news/notice/list 에 있는 게시글들을 크롤링해 와서
# #### nlp 알고리즘을 이용하여 각 게시글들의 내용만 보고 
# #### 게시글들을 군집화 할 수 있는지 실험한다. 
# 
# ## scenario
#  1.  게시글 크롤링
#  2.  크롤링 한 데이터 전처리
#  3.  tokenizing
#  4.  Doc2Vec
#  5.  Clustering

# In[1]:


import time
from time import sleep
from tqdm import tqdm
import numpy as np
import pandas as pd


# In[2]:


import pickle


# In[3]:


from konlpy.tag import *
from selenium import webdriver
import csv
import re
import selenium 
from selenium import webdriver
import random
from time import sleep
import re
from random import randint
import requests
from bs4 import BeautifulSoup as bs
import json
from urllib.request import urlopen
import os


# ## 1. 게시글 Crawling & 2. 크롤링 한 데이터 전처리 &  3. tokenizing

# - JavaScript로 Render되는 web page crawling 
# - Ref : http://theautomatic.net/2019/01/19/scraping-data-from-javascript-webpage-python/  (참고만 했습니다)
# - Ref : https://stackoverflow.com/questions/8049520/web-scraping-javascript-page-with-python (코드 활용했습니다)

# - Korean Stopwords Ref : https://www.ranks.nl/stopwords/korean
# - 불용어를 stopwords에 저장합니다.

# In[6]:


#여러번 다시 하러면 너무 오래걸리니까 pickle로 저장했습니다. 
with open('crawled_list.pickle', 'rb') as f:
    crawled_list = pickle.load(f)
with open('title_list.pickle', 'rb') as f:
    title_list = pickle.load(f)


# In[7]:


stopwords=[]
with open('stopwords.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        for i in row:
            stopwords.extend(i.split(','))


# 불용어 이거나 한글이 아닌 문자 삭제

# In[8]:


def filtering(inputstr):
    inputstr=re.sub('[^가-힣]','',inputstr)
    if inputstr=='':
        return -1
    if inputstr in stopwords:
        return -1
    return inputstr


# crawling 된 text, filtering 후 형태소 추출

# In[9]:


def webCrawler(notice):
    driver = webdriver.PhantomJS()
    my_url='https://cs.skku.edu/news/notice/view/'+str(notice)
    driver.get(my_url)
    sleep(random.uniform(2,3))
    post_text=driver.find_element_by_id("text")
    title_text=driver.find_element_by_id("title")
    crawled_text=post_text.text
    title=title_text.text
    twitter = Twitter()
    morphs_notice=twitter.morphs(crawled_text)
    text_filtered = []
    for i in morphs_notice:
        inputstr=filtering(i)
        if inputstr!=-1:
            text_filtered.append(inputstr)
    return text_filtered, title


# - 크롤링하고자 하는 게시글의 범위를 입력 받아서, 
# - 필터된 게시글들의 형태소들을 반환합니다. 

# In[10]:


def webCrawler_list(start, end):
    morphs_list=[]
    title_list=[]
    for i in tqdm(range(start,end+1)):
        text, title=webCrawler(i)
        morphs_list.append(text)
        title_list.append(title)
    return morphs_list, title_list


# In[11]:


startNotice=int(4300)
endNotice=int(4480)
size=len(range(startNotice,endNotice+1))


# In[12]:


# crawled_list, title_list =webCrawler_list(startNotice,endNotice)


# In[13]:


with open('crawled_list.pickle', 'wb') as f:
    pickle.dump(crawled_list, f, pickle.HIGHEST_PROTOCOL)


# In[14]:


with open('title_list.pickle', 'wb') as f:
    pickle.dump(title_list, f, pickle.HIGHEST_PROTOCOL)


# In[18]:


#예시
crawled_list[1]


# In[23]:


d={'id':range(size),'morphs':crawled_list, 'title': title_list}


# In[24]:


df=pd.DataFrame(data=d)


# In[34]:


df.head()


# ##  4.  Doc2Vec

# - 각 게시글의 형태소들을 vector화 합니다.(Doc2Vec)

# In[37]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['morphs'])]


# In[38]:


model = Doc2Vec(documents, size=25, window=2, min_count=1, workers=4)


# In[39]:


#appending all the vectors in a list for training
X=[]
for i in range(size):
    X.append(model.docvecs[i])
X[:3]


#  Doc2Vec Ref : https://medium.com/@japneet121/document-vectorization-301b06a041

# ## 5.  Clustering

# Vector화 한 Document들을 군집화 하는 과정입니다.

# ### 계층적 군집화

# In[40]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)


# In[41]:


clusters={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters2={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters_num={0:[],1:[],2:[],3:[],4:[],5:[]}
for i in range(size):
    clusters[cluster.labels_[i]].append(' '.join(df.ix[i,'morphs']))
    clusters2[cluster.labels_[i]].append(df.ix[i,'title'])
    clusters_num[cluster.labels_[i]].append(i+startNotice)


# In[42]:


clusters2[0]


# - 공백인 문자열들을 얼마나 잘 군집화 시켰는지.

# In[43]:


blankNum=np.zeros(6)
for i in range(6):
    print('set' + str(i))
    for j in range(len(clusters[i])):
        if clusters2[i][j]!='':
            blankNum[i]+=1
    print(blankNum[i])


# - 군집들의 제목들 보기

# In[44]:


for i in range(6):
    print('set'+str(i))
    for j in range(len(clusters2[i])):
        if clusters2[i][j]!='':
            print(clusters2[i][j])
    print('======================================================')


# ### k-means 군집화

# In[78]:


#import the modules
from sklearn.cluster import KMeans
import numpy as np
#create the kmeans object withe vectors created previously
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
#print all the labels
print (kmeans.labels_)
#craete a dictionary to get cluster data
clusters={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters2={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters_num={0:[],1:[],2:[],3:[],4:[],5:[]}
for i in range(size):
    clusters[kmeans.labels_[i]].append(' '.join(df.ix[i,'morphs']))
    clusters2[kmeans.labels_[i]].append(df.ix[i,'title'])
    clusters_num[kmeans.labels_[i]].append(i+startNotice)


# - 공백인 문자열들을 얼마나 잘 군집화 시켰는지.

# In[53]:


blankNum=np.zeros(6)
for i in range(6):
    print('set' + str(i))
    for j in range(len(clusters[i])):
        if clusters2[i][j]!='':
            blankNum[i]+=1
    print(blankNum[i])


# - 공백인 문자열들을 군집화하는 것은 kmeans가 더 아쉬운 결과를 보여줍니다.

# - 군집들의 제목들 보기

# In[80]:


for i in range(6):
    print('set'+str(i))
    for j in range(len(clusters2[i])):
        if clusters2[i][j]!='':
            print(clusters2[i][j])
    print('======================================================')


# ### 개선 방향 
# - 사진만 게시되어있는 글도 많고, 
# - 군집화된 글들의 제목으로 봤을때 잘 묶인거 같지가 않습니다.
# - 게시글 내용들을 기준으로 하지 말고, 제목을 vectorizing해서 진행해보겠습니다.

# - 게시글의 title을 대상으로 형태소 추출, 필터링을 진행합니다.

# In[45]:


title_list2=[]
for i in title_list:
    ret = filtering(i)
    if ret == -1:
        title_list2.append('')
    else:
        title_list2.append(ret)


# In[46]:


title_list2


# In[47]:


df['title2']=title_list2


# - 형태소 추출한 list를 대상으로 vectorizing을 진행합니다.

# In[48]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(title_list2)]


# In[49]:


model = Doc2Vec(documents, size=25, window=2, min_count=1, workers=4)


# In[50]:


X=[]
for i in range(size):
    X.append(model.docvecs[i])
print(X[:3])


# In[51]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

clusters={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters2={0:[],1:[],2:[],3:[],4:[],5:[]}
clusters_num={0:[],1:[],2:[],3:[],4:[],5:[]}
for i in range(size):
    clusters[cluster.labels_[i]].append(' '.join(df.ix[i,'morphs']))
    clusters2[cluster.labels_[i]].append(df.ix[i,'title'])
    clusters_num[cluster.labels_[i]].append(i+startNotice)


# In[52]:


for i in range(6):
    print('set'+str(i))
    for j in range(len(clusters2[i])):
        if clusters2[i][j]!='':
            print(clusters2[i][j])
    print('======================================================')


# ### 느낀점.
# - 사실 이렇게까지 했는데도 잘 군집이 됐는지 모르겠습니다.
# - 공부하는 중 tf-idf 라는 개념을 알게 되었는데
# - 이 개념을 적용하면 게시글 제목의 키워드를 뽑아내어
# - 잘 군집화 되었는지 알 수 있을 것 같습니다.
# 

# In[ ]:




