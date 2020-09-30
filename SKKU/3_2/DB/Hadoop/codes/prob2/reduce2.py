#!/usr/bin/env python
import sys

cords=[]
for key_value in sys.stdin:
    key_value = key_value.strip()
    key, table_value = key_value.split("\t")
    values = table_value.split(',')
    cords.append([values[0],int(values[1]),int(values[2])])
ret=[]
for idx in range(len(cords)):
    is_Dominant=True
    for idx2 in range(len(cords)):
        if idx == idx2:
            continue
        else:
            if cords[idx2][1]<cords[idx][1] and cords[idx2][2]<cords[idx][2]:
                is_Dominant=False
    if is_Dominant:
        ret.append(cords[idx][0])
for ret_name in ret:
    print(ret_name)