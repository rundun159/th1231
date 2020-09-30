#!/usr/bin/env python
import sys

last_deptno = None
last_dname = None
last_campus = None
gpa_list = []
for key_value in sys.stdin:
    key_value = key_value.strip()
    deptno, table_value = key_value.split("\t")

    table, values = table_value.split(',')[0], table_value.split(',')[1:]
    if last_deptno == deptno:
        if table=='student':
            gpa_list.append(float(values[0]))
        else:
            last_dname = values[0]
            last_campus= values[1]
    else:
        if last_deptno and last_dname and last_campus and gpa_list:
            gpa_sum=0
            student_no=0
            max_gpa=-1
            for gpa in gpa_list:
                gpa_sum+=gpa
                student_no+=1
                if(gpa>max_gpa):
                    max_gpa=gpa

            avg_gpa=gpa_sum/student_no
            if(avg_gpa>3.5):
                print("%s,%s,%s"%(last_dname,str(max_gpa),last_campus))
            gpa_list=[]
            last_deptno = None
            last_dname = None
            last_campus = None
            
        if table=='student':
            gpa_list=[float(values[0])]
        else:
            last_dname=values[0]
            last_campus=values[1]
        last_deptno = deptno

if last_deptno and last_dname and last_campus and gpa_list:
    gpa_sum=0
    student_no=0
    max_gpa=-1
    for gpa in gpa_list:
        gpa_sum+=gpa
        student_no+=1
        if(gpa>max_gpa):
            max_gpa=gpa
    avg_gpa=gpa_sum/student_no
    if(avg_gpa>3.5):
        print("%s, %s,%s"%(last_dname,str(max_gpa),last_campus))
        gpa_list=[]
