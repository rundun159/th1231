#!/usr/bin/env python

import sys

for line in sys.stdin:
    line=line.strip()
    tuple_list = line.split(",")
    if len(tuple_list)==6: # 'student' table
        deptno = int(tuple_list[4])
        gpa = tuple_list[5]
        print('{0}\t{1}'.format(str(deptno),'student,'+gpa))
    else: # 'dept' table
        deptno = int(tuple_list[0])
        dname = tuple_list[1]
        campus = tuple_list[2]
        print('{0}\t{1}'.format(str(deptno),'dept,'+dname+','+campus))
