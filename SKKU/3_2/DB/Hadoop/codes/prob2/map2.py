#!/usr/bin/env python

import sys

for line in sys.stdin:
	line=line.strip()
	tuple_list = line.split(",")
	if len(tuple_list)==3:
		print('{0}\t{1}'.format(str(1),tuple_list[0]+','+tuple_list[1]+','+tuple_list[2]))