import numpy as np
import random
import os
from tqdm import tqdm
from datetime import datetime
make_max_input = True
make_file = 20
p = 0.3
max_list = [False]*make_file
for i in range(make_file):
    if random.random() < p:
        max_list[i] = True
random.seed(datetime.now())

MIN_N = 1000
MAX_N = 1000000
MIN_T = 1
MAX_T = 10
MIN_M = 1
MAX_M = 15


USABLE=[' ']
for i in range(0,10):
    USABLE+=str(i)
for i in range(ord('A'),ord('Z')+1):
    USABLE+=chr(i)
for i in range(ord('a'),ord('z')+1):
    USABLE+=chr(i)
LEN_USABLE = len(USABLE)

for file_idx in range(make_file):
    ATTRIBUTE_LIST=['P_ID','LastName','LASTNAME','FirstName','FIRSTNAME','Address','ADDRESS','City','CITY','LOCATION']
    if max_list[file_idx]:
        n = MAX_N
        t = MAX_T
    else:
        n = random.randint(MIN_N,MAX_N)
        t = random.randint(MIN_T,MAX_T)
    key_t = random.randint(1,t)

    random.shuffle(ATTRIBUTE_LIST)
    ATTRIBUTE_LIST[key_t-1]+='(Key)'

    try:
        os.mkdir('inputs')
    except:
        print("inputs directory already exists")
    file_list = os.listdir('inputs')
    last_idx = -1
    for f in file_list:
        idx=int(f.split('_')[1].split('.')[0])
        if idx>last_idx:
            last_idx=idx

    print(n,t,key_t)

    with open('./inputs/input_'+str(last_idx+1)+'.txt','w') as f:
        if max_list[file_idx]:
            print('input_'+str(last_idx+1)+'.txt is max file')
        f.write(str(n))
        f.write('\n$\n')
        for i in range(t-1):
            f.write(ATTRIBUTE_LIST[i])
            f.write(' : ')
        f.write(ATTRIBUTE_LIST[t-1])
        f.write('\n$\n')
        for i in tqdm(range(n)):
            for t_ in range(t):
                if max_list[file_idx]:
                    m = MAX_M
                else:
                    m = random.randint(MIN_M,MAX_M)
                for m_ in range(m):
                    char_idx = random.randint(0,LEN_USABLE-1)
                    f.write(USABLE[char_idx])
                if t_ != t-1:
                    f.write(':')
            if i != n-1:
                f.write('\n')