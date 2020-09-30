# -*- coding: utf-8 -*-

import json # import json module
import numpy as np
import csv
import pickle
import math
import codecs
import copy
from tqdm import tqdm

def stopWords():
    stopwords=[]
    stopwords_dict={}
    with open('stopwords.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            for i in row:
                stopwords.extend(i.split(','))
    for words in stopwords:
        stopwords_dict[words] = 1
    return stopwords_dict

def ret_dict_list(json_data,stopwords_dict): #stopwords_dict(불용어)에 있는 단어들을 제외하면서, json data에 있는 doc들의 tf 계산하기
    dict_list = []
    for i in range(len(json_data)):
        dict_list.append({})
    for doc_num in range(len(json_data)):
        dict_doc = dict_list[doc_num]
        sentence = json_data[doc_num]['sentence']
        for i in sentence:
            before_NL = i.split('\n')[0]
            if before_NL == '':
                continue
            sentence = before_NL.split(' ')
            for morphs in sentence:
                morph_vector = morphs.split('+')
                for morphs in morph_vector:
                    morph = morphs.split('/')
                    if len(morph) == 2:
                        # if morph[1][0] == 'N' or morph[1][0] == 'V':
                        if morph[1][0] == 'N':
                            if not (morph[0] in stopwords_dict):
                                if morph[0] in dict_doc:
                                    pass
                                    dict_doc[morph[0]] += 1
                                else:
                                    dict_doc[morph[0]] = 1
    # print("바뀜")
    return dict_list

#train_labels 만드는 함수
def ret_labels(json_data):
    empty_list=[]
    labels=np.zeros(len(json_data))
    for idx in range(len(json_data)):
        try:
            label=int(json_data[idx]['answer'])
        except:
            empty_list.append(idx)
            label = np.random.randint(1, 6)
        labels[idx]=label
    train_labels_encoded = np.zeros(shape=(len(labels), 5))
    for i in range(len(labels)):
        train_labels_encoded[i][int(labels[i] - 1)] = 1
    return train_labels_encoded, empty_list

def ret_corpus_dict(dict_list): #key: corpus의 단어들 / value : 각 단어딀의 df
    corpus_dict = {}
    for doc_dict in dict_list:
        for key in doc_dict:
            if key in corpus_dict:
                corpus_dict[key] += 1
            else:
                corpus_dict[key] = 1
    return corpus_dict

def ret_zeros_corpus_dict(corpus_dict):
    zeros_corpus_dict = copy.deepcopy(corpus_dict)
    for keys in zeros_corpus_dict:
        zeros_corpus_dict[keys]=0
    return zeros_corpus_dict
def ret_zeros_corpus_np(corpus_np):
    zeros_corpus_np=copy.deepcopy(corpus_np)
    zeros_corpus_np[:]['value'] = 0
    return zeros_corpus_np

def ret_doc_tf_idf(corpus_dict,corpus_np,dict_list): # doc의 tf list(dict_list)를 전달 받아서 tf idf를 계산해서 return #corpus_dict는 corpus에 등장하는 단어들.dict 형태.
    doc_tf_idf_list=[]                              #corpus_np는 corpus의 단어들의 idf. numpy형태.
    empty_list=[] # 단어가 아무것도 없는 doc의 index 저장
    zero_corpus_dict=ret_zeros_corpus_dict(corpus_dict)
    zero_corpus_np=ret_zeros_corpus_np(corpus_np)
    # for i in range(len(dict_list)):
    for i in tqdm(range(len(dict_list)), mininterval=1):
        sum=0
        doc_dict=dict_list[i]
        if len(doc_dict)==0:
            empty_list.append(i)
        doc_tf_dict=copy.deepcopy(zero_corpus_dict)
        doc_tf_idf_np=copy.deepcopy(zero_corpus_np)
        for key, value in doc_dict.items():
            if key in doc_tf_dict:
                doc_tf_dict[key]=value #각 문서에 대해서 corpus에 있는 단어들 대상으로 tf를 저장하는 과정
        for idx in range(len(corpus_np)):
            doc_tf_idf_np[idx]['value']=corpus_np[idx]['value']*doc_tf_dict[corpus_np[idx]['key']] #앞의 op : 해당 단어의 idf , 뒤의 op : 해당 단어의 tf
            sum+=doc_tf_idf_np[idx]['value']
        if sum==0:
            doc_tf_idf_np[:]['value']=0
        else:
            doc_tf_idf_np[:]['value']/=sum
        doc_tf_idf_list.append(copy.deepcopy(doc_tf_idf_np['value']))
        del doc_tf_dict
        del doc_tf_idf_np
    return doc_tf_idf_list, empty_list

def ret_doc_tf_idf_test(corpus_dict,corpus_np,dict_list): # doc의 tf list(dict_list)를 전달 받아서 tf idf를 계산해서 return #corpus_dict는 corpus에 등장하는 단어들.dict 형태.
    doc_tf_idf_list=[]                              #corpus_np는 corpus의 단어들의 idf. numpy형태.
    zero_corpus_dict=ret_zeros_corpus_dict(corpus_dict)
    zero_corpus_np=ret_zeros_corpus_np(corpus_np)
    # for i in range(len(dict_list)):
    for i in tqdm(range(len(dict_list)), mininterval=1):
        sum=0
        doc_dict=dict_list[i]
        doc_tf_dict=copy.deepcopy(zero_corpus_dict)
        doc_tf_idf_np=copy.deepcopy(zero_corpus_np)
        for key, value in doc_dict.items():
            if key in doc_tf_dict:
                doc_tf_dict[key]=value #각 문서에 대해서 corpus에 있는 단어들 대상으로 tf를 저장하는 과정
        for idx in range(len(corpus_np)):
            doc_tf_idf_np[idx]['value']=corpus_np[idx]['value']*doc_tf_dict[corpus_np[idx]['key']] #앞의 op : 해당 단어의 idf , 뒤의 op : 해당 단어의 tf
            sum+=doc_tf_idf_np[idx]['value']
        if sum==0:
            doc_tf_idf_np[:]['value']=0
        else:
            doc_tf_idf_np[:]['value']/=sum
        doc_tf_idf_list.append(copy.deepcopy(doc_tf_idf_np['value']))
        del doc_tf_dict
        del doc_tf_idf_np
    return doc_tf_idf_list

def ret_corpus_np(corpus_dict,doc_num):       #corpus에 있는 단어들의 idf 계산하기. doc_num: 문서의 총 갯수.
    corpus_np = np.empty(shape=len(corpus_dict), dtype=[('key', 'object'), ('value', 'f8')])
    N = doc_num
    idx = 0
    for key, val in corpus_dict.items():
        corpus_np[idx]['key'] = key
        corpus_np[idx]['value'] = math.log(N / val, 2)
        idx += 1
    return corpus_np
def ret_tf_idf_np(corpus_np,dict_list):
    pass

    #각 doc들의 tf가 저장된 dict가 주어졌을때, tf-idf를 numpy 형태로 반환하기
