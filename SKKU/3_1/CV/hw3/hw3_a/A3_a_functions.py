import numpy as np
import cv2
import sys
import time
import math
from scipy.spatial import distance
import pickle
import A2_functions as fn
from tqdm  import tqdm
initial_T=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
def normalize(points):
    moved_points=np.copy(points)
    T1=np.copy(initial_T)
    T2=np.copy(initial_T)
    T1[0,2]=-np.mean(moved_points,axis=0)[0]
    T1[1,2]=-np.mean(moved_points,axis=0)[1]
    moved_points[:,0]=moved_points[:,0]-np.mean(points,axis=0)[0]
    moved_points[:,1]=moved_points[:,1]-np.mean(points,axis=0)[1]
    max_x=np.max(np.absolute(moved_points),axis=0)[0]
    max_y=np.max(np.absolute(moved_points), axis=0)[1]
    T2[0,0]=1/max_x
    T2[1,1]=1/max_y
    moved_points[:,0]=moved_points[:,0]/max_x
    moved_points[:,1]=moved_points[:,1]/max_y
    T2=T2.dot(T1)
    del T1
    return T2, moved_points
def compute_F_norm ( srcP , destP ):
    T_s,norm_srcP=normalize(srcP)
    T_d,norm_destP=normalize(destP)
    A=np.ones(shape=(len(srcP),9),dtype=float)
    for i in range(len(srcP)):
        A[i][0:3]*=norm_srcP[i][0]
        A[i][3:6]*=norm_srcP[i][1]
        A[i,[0,3,6]]*=norm_destP[i][0]
        A[i,[1,4,7]]*=norm_destP[i][1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v=np.transpose(vh)
    F=v[:,8]
    try:
        F/=F[8]
    except:
        F/=0.00000000001
    F=np.reshape(F,(3,3))
    fromS_toP=np.transpose(T_s).dot(F).dot(T_d)
    return fromS_toP
def ret_First_Ransac(srcP,destP,pick=8):
    rand_Idx =np.asarray(len(srcP) * np.random.rand(pick), dtype=np.int)
    pickedSrcP=srcP[rand_Idx]
    pickedDestP=destP[rand_Idx]
    F=compute_F_norm(pickedSrcP,pickedDestP)
    #Make F's Rank 2
    U, s, V = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F2 = np.dot(U, np.dot(np.diag(s), V))
    F2/= F2[2, 2]
    return F , F2
def getLines(srcP,destP,F2):
    lines = np.zeros((len(srcP), 6), np.float)
    for i in range(len(srcP)):
        match=np.array([
            [srcP[i][0],destP[i][0]],
            [srcP[i][1],destP[i][1]],
            [1, 1],
        ])
        line_temp=np.dot(F2,match)
        lines[i,0:3]=line_temp[:,0].T
        lines[i,3:6]=line_temp[:,1].T
    return lines
def get_cnt_inliners(srcP,destP,lines):
    cnt=0
    inliners=[]
    for idx in range(len(srcP)):
        srcP_T=np.array([
            srcP[idx][0],
            srcP[idx][1],
            1
        ])
        multiplied=np.dot(srcP_T.T,lines[idx,0:3])
        if(int(multiplied)==0):
            inliners.append(idx)
            cnt+=1
    return cnt, inliners
def compute_F_mine (srcP, destP,pick=20):
    F,F2=ret_First_Ransac(srcP,destP,pick)
    lines=getLines(srcP,destP,F2)
    cnt,inliners=get_cnt_inliners(srcP,destP,lines)
    num=0
    while cnt!=len(srcP):
        if(num>100):
            return F
        if(cnt<30):
            F, F2 = ret_First_Ransac(srcP, destP, pick)
            lines = getLines(srcP, destP, F2)
            cnt, inliners = get_cnt_inliners(srcP, destP, lines)
        else:
            inline_srcP=srcP[inliners]
            inline_destP=destP[inliners]
            new_F,new_F2=ret_First_Ransac(inline_srcP,inline_destP,pick)
            new_lines=getLines(srcP,destP,new_F2)
            new_cnt,new_inliners=get_cnt_inliners(srcP,destP,new_lines)
            if(new_cnt>cnt):
                F=new_F
                F2=new_F2
                cnt=new_cnt
                inliners=new_inliners
    return F

def compute_F_raw ( srcP , destP ):
    A=np.ones(shape=(len(srcP),9),dtype=float)
    for i in range(len(srcP)):
        A[i][0:3]*=srcP[i][0]
        A[i][3:6]*=srcP[i][1]
        A[i,[0,3,6]]*=destP[i][0]
        A[i,[1,4,7]]*=destP[i][1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v=np.transpose(vh)
    F=v[:,8]
    try:
        F/=F[8]
    except:
        F/=0.00000000001
    F=np.reshape(F,(3,3))
    return F
def compute_F_min(srcP , detP):
    pass

#ransac 할때 알아야 할것 :
#직선으로부터 (직선을 정의하는것도 뭔지 잘 모르겠음)
#점까지의 거리를 구하는 방법
