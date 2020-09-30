```python
import time
from time import sleep
from tqdm import tqdm
import numpy as np
import pandas as pd
```


```python
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
```


```python
def retMoviePd(year):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; InteSl Mac O X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    main_url = 'https://movie.daum.net/boxoffice/yearly?year='+str(year) 

    data = []
    netizen=[]
    critic=[]
    view=[]
    url=[]
    movieId=[]
    cnt= 0 

    response = requests.get(main_url,headers=headers)
    body_table = bs(response.text,'html.parser')

    poster_table = body_table.findAll('div',{"class":"wrap_movie"})

        #영화 이름, 네티즌 평점, 전문가 평점을 얻는 코드입니다. 
    for i in tqdm(range(len(poster_table)),desc=str(year)):
        poster=poster_table[i]
        title_tag= poster.find('a',{"class":"name_movie #title"})
        
        data.append(title_tag.text)                                 
        url.append('https://movie.daum.net/'+str(title_tag['href']))
        movieId.append(int(url[i].split('=')[1]))
        score_tag=poster.find('span',{"class":"wrap_grade grade_netizen"})
            #평점을 가지고 있는 span의 클래스가 일정한 형식을 가지고 있어서
            #정형식을 이용했습니다.
        netizen_scores=score_tag.findAll('span',{"class":re.compile("num_grade num_.*.*")})
        netizen_score=int(netizen_scores[0].text)+int(netizen_scores[1].text)/10
        netizen.append(netizen_score)

            #평론가 평점이 없는 영화를 위해 예외 처리를 했습니다.
        try:
            score_tag=poster.find('span',{"class":"wrap_grade grade_critic"})
            critic_scores=score_tag.findAll('span',{"class":re.compile("num_grade num_.*.*")})
            critic_score=int(critic_scores[0].text)+int(critic_scores[1].text)/10
            critic.append(critic_score)
        except Exception as e:
            critic.append(None)
            cnt+=1
        #누적 관람객 정보를 가지고 있는 json 파일을 열람합니다. 
        viewUrl = 'https://movie.daum.net/moviedb/main/totalAudience.json?movieId='+str(movieId[i])
        json_string=requests.get(viewUrl).text
        viewDict=json.loads(json_string)
        view.append(viewDict["totalAudience"])

    index=range(len(data))
    columns=['title','netizen','ciritic','views']    
    infopd=pd.DataFrame(index=index,columns=columns)
    infopd['title']=data
    infopd['netizen']=netizen
    infopd['ciritic']=critic
    infopd['views']=view
    infopd['MovieId']=movieId
    infopd.set_index('MovieId',inplace = True)
    return infopd
```


```python
def retInfoYears(startYear,endYear):
    ret = []
    for year in range(startYear,endYear+1):
        ret.append(retMoviePd(year))
    return ret
```


```python
def pdTocsv(infopd,title):
    path = os.getcwd()+"/movies"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    infopd.to_csv(path+'/'+title, sep=',',na_rep='NaN',encoding='utf-8-sig')
    print(path+'/'+title)
```


```python
def inputYears():
    print("시작 연도")
    startYear=int(input())
    print("마지막 연도")    
    endYear=int(input())
    ret=[startYear,endYear]
    return ret
```


```python
def main():
    years=inputYears()
    infoPdList = retInfoYears(years[0],years[1])
    for i in range(len(infoPdList)):
        pdTocsv(infoPdList[i],str(i+years[0])+'.csv')
```


```python
main()
```

    시작 연도
    2007
    마지막 연도
    2007


    2007: 100%|████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.34it/s]


    Creation of the directory C:\Users\USER\Drive\TH\Study\ToBigs\BIgData\Week6\크롤링\수업자료/movies failed
    C:\Users\USER\Drive\TH\Study\ToBigs\BIgData\Week6\크롤링\수업자료/movies/2007.csv



```python

```
