import numpy as np
import math
import pickle
import struct
NUM_CLUSTER=8
def extractSIFT():
    SIFTdes=np.empty(1000,dtype=np)
    for i in range(1000):
        idx=100000+i
        des = np.fromfile("./des/sift"+str(idx), dtype='uint8')
        des = np.reshape(des, (int(len(des) / 128), 128))
        SIFTdes[i]=des
    return SIFTdes
def combine_Des(SIFT_Des):
    total_Des = SIFT_Des[0]
    for idx in range(1,len(SIFT_Des)):
        total_Des=np.concatenate((total_Des,SIFT_Des[idx]))
    return total_Des
def L1Dist(a,b):
    return np.sum(np.absolute(a-b))
def L2Dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))
def K_means(total_Des,num=NUM_CLUSTER):   #number of centers are passed . Gonna be put only 10K Des
    randIdx = np.asarray(len(total_Des) * np.random.rand(num), dtype=np.int)  #pick randomly 8 items among 10K
    center_points=total_Des[randIdx]
    count=0
    while 1:
        print(count)
        new_center_idx=np.empty(shape=num,dtype=np)
        for i in range(len(total_Des)):
            minDist=987654321.0
            minIdx=-1
            for classes in range(num):
                nowDist=L2Dist(center_points[classes],total_Des[i])
                if(minDist>nowDist):
                    minDist=nowDist
                    minIdx=classes
            new_center_idx[minIdx]=np.append(new_center_idx[minIdx],i)
        new_center_points=retNewCenters(center_points,new_center_idx,total_Des,num)
        dist=np.sum(np.absolute(new_center_points-center_points))
        if(dist<0.5 or count>500):
            with open('./pdata/center_points.pickle', 'wb') as f:
                pickle.dump(new_center_points, f, pickle.HIGHEST_PROTOCOL)
            return new_center_points
        center_points=new_center_points
        count+=1
        print("It moved for ",end=' ')
        print(dist)
        print("Noew variance is ", end=' ')
        print(math.log(variance_From_centers(new_center_points,new_center_idx,total_Des,num),2))
        with open('./pdata/center_points.pickle', 'wb') as f:
            pickle.dump(new_center_points, f, pickle.HIGHEST_PROTOCOL)
        print("Saved")
def small_K_means(total_Des,num=NUM_CLUSTER):   #number of centers are passed . Gonna be put only 10K Des
    randIdx = np.asarray(len(total_Des) * np.random.rand(num), dtype=np.int)  #pick randomly 8 items among 10K
    center_points=total_Des[randIdx]
    count=0
    while 1:
        # print(count, end=' ')
        new_center_idx=np.empty(shape=num,dtype=np)
        for i in range(len(total_Des)):
            minDist=987654321.0
            minIdx=-1
            for classes in range(num):
                nowDist=L2Dist(center_points[classes],total_Des[i])
                if(minDist>nowDist):
                    minDist=nowDist
                    minIdx=classes
            new_center_idx[minIdx]=np.append(new_center_idx[minIdx],i)
        new_center_points=retNewCenters(center_points,new_center_idx,total_Des,num)
        dist=np.sum(np.absolute(new_center_points-center_points))
        if(dist==0 or count>50):
           return new_center_points
        center_points=new_center_points
        count+=1
        # print("It moved for ",end=' ')
        # print(dist , end=' ')
        # print("Noew variance is ", end=' ')
        # print(math.log(variance_From_centers(new_center_points,new_center_idx,total_Des,num),2))

def variance_From_centers(center_points,center_idx,total_Des,num):
    ret=0
    for classes in range(num):
        for idx in range(len(center_idx[classes])):
            ret+=L2Dist(center_points[classes],total_Des[center_idx[classes][idx]])
    return ret

def retNewCenters(center_points,new_center_idx,total_Des,num):
    new_center_points=np.zeros(shape=(num,128),dtype=np.float32)
    for classes in range(num):
        new_center_idx[classes]=np.delete(new_center_idx[classes],0)
        if len(new_center_idx[classes])==0:
            new_center_points[classes]=center_points[classes]
        else:
            new_center_points[classes]=np.sum(total_Des[np.asarray(new_center_idx[classes],dtype=np.int)],axis=0,dtype=np.float32)/len(new_center_idx[classes])
    return new_center_points

def main():
    try:
        with open('./pdata/SIFT_Des.pickle', 'rb') as f:
            SIFT_Des=extractSIFT()
    except:
        SIFT_Des=extractSIFT()
        with open('./pdata/SIFT_Des.pickle', 'wb') as f:
            pickle.dump(SIFT_Des, f, pickle.HIGHEST_PROTOCOL)

    try:
        with open('./pdata/total_Des.pickle', 'rb') as f:
            total_Des=pickle.load(f)
    except:
        total_Des=combine_Des(SIFT_Des)
        with open('./pdata/total_Des.pickle', 'wb') as f:
            pickle.dump(total_Des, f, pickle.HIGHEST_PROTOCOL)


    try:
        with open('./pdata/Centers_Des.pickle', 'rb') as f:
            total_Des=pickle.load(f)
    except:
        total_Des = retCenterSet(SIFT_Des, 8)
        with open('./pdata/Centers_Des.pickle', 'wb') as f:
            pickle.dump(total_Des, f, pickle.HIGHEST_PROTOCOL)
    # total_Des=selectSpecimen(SIFT_Des,200)
    # print(total_Des.shape)
    # randIdx = np.asarray(len(total_Des) * np.random.rand(100000), dtype=np.int)  #pick randomly 8 items among 10K
    # total_Des=total_Des[randIdx]

    # try:
    #     with open('./pdata/center_points.pickle', 'rb') as f:
    #         center_points=pickle.load(f)
    # except:
    #     center_points=K_means(total_Des,NUM_CLUSTER)
    #
    try:
        with open('./pdata/my_Des.pickle', 'rb') as f:
            my_Des=pickle.load(f)
    except:
        my_Des = make_my_Des_using_8(total_Des)
        with open('./pdata/my_Des.pickle', 'wb') as f:
            pickle.dump(my_Des, f, pickle.HIGHEST_PROTOCOL)
    A3_2015313754 = open("./eval/A3_2015313754.des", "wb")
    data=struct.pack('i',1000)
    A3_2015313754.write(data)
    data=struct.pack('i',1024)
    A3_2015313754.write(data)
    for imgIdx in range(1000):
        for fIdx in range(1024):
            data=struct.pack('f',my_Des[imgIdx][fIdx])
            A3_2015313754.write(data)
    return

#    SIFT_Des   0~1000 exists
def make_my_Des_using_8(total_Des):
    myDes=np.zeros(shape=(1000,1024),dtype=np.float32)
    for i in range(len(total_Des)):
        for j in range(8):
            myDes[i,j*128:(j+1)*128]=total_Des[i*8+j]
    return myDes
def make_my_Des_For_16(SIFT_Des,center_points,num=NUM_CLUSTER):
    myDes=np.zeros(shape=(1000,1024),dtype=np.float32)
    for imgIdx in range(len(SIFT_Des)):
        center_idx = np.empty(shape=NUM_CLUSTER, dtype=np)
        for desIdx in range(len(SIFT_Des[imgIdx])):
            minDist = 987654321.0
            minIdx = -1
            for classes in range(num):
                nowDist = L2Dist(center_points[classes], SIFT_Des[imgIdx][desIdx])
                if (minDist > nowDist):
                    minDist = nowDist
                    minIdx = classes
            center_idx[minIdx]=np.append(center_idx[minIdx],desIdx)
        for classes in range(num):
            center_idx[classes] = np.delete(center_idx[classes], 0)
            if len(center_idx[classes]) == 0:
                myDes[imgIdx,classes*128:(classes+1)*128]=0
            else:
                # diff=center_points[classes]*len(center_idx[classes])
                # diff-=np.sum(SIFT_Des[imgIdx][np.asarray(center_idx[classes],dtype=np.int)],axis=0)
                diff=center_points[classes]-SIFT_Des[imgIdx][int(center_idx[classes][0])]
                for idx in range(1,len(center_idx[classes])):
                    diff += center_points[classes] - SIFT_Des[imgIdx][int(center_idx[classes][idx])]
                diff/=len(center_idx[classes])
                diff2=np.zeros(shape=64,dtype=np.float)
                for i in range(64):
                    diff2[i]=(diff[i*2]+diff[i*2+1])/2
                myDes[imgIdx,classes*64:(classes+1)*64]=diff2
    return myDes
def make_my_Des(SIFT_Des,center_points,num=NUM_CLUSTER):
    myDes=np.zeros(shape=(1000,1024),dtype=np.float32)
    for imgIdx in range(len(SIFT_Des)):
        center_idx = np.empty(shape=NUM_CLUSTER, dtype=np)
        for desIdx in range(len(SIFT_Des[imgIdx])):
            minDist = 987654321.0
            minIdx = -1
            for classes in range(num):
                nowDist = L2Dist(center_points[classes], SIFT_Des[imgIdx][desIdx])
                if (minDist > nowDist):
                    minDist = nowDist
                    minIdx = classes
            center_idx[minIdx]=np.append(center_idx[minIdx],desIdx)
        for classes in range(num):
            center_idx[classes] = np.delete(center_idx[classes], 0)
            if len(center_idx[classes]) == 0:
                myDes[imgIdx,classes*128:(classes+1)*128]=0
            else:
                # diff=center_points[classes]*len(center_idx[classes])
                # diff-=np.sum(SIFT_Des[imgIdx][np.asarray(center_idx[classes],dtype=np.int)],axis=0)
                diff=center_points[classes]-SIFT_Des[imgIdx][int(center_idx[classes][0])]
                for idx in range(1,len(center_idx[classes])):
                    diff += center_points[classes] - SIFT_Des[imgIdx][int(center_idx[classes][idx])]
                diff/=len(center_idx[classes])
                myDes[imgIdx,classes*128:(classes+1)*128]=diff
    return myDes
def retCenterSet(SIFT_Des,num=100):
    center_set=retSmallCenter(SIFT_Des[0],num)
    for imgIdx in range(1,len(SIFT_Des)):
        print("Imge Idx : ",end='')
        print(imgIdx)
        small_center=retSmallCenter(SIFT_Des[imgIdx],num)
        center_set=np.concatenate((center_set,small_center),axis=0)
    return center_set
def retSmallCenter(SIFT_Des_Img,num=100):
    center_points=small_K_means(SIFT_Des_Img,num)
    return center_points
def selectInsiders_Set(SIFT_Des,num1=10,num2=100):
    Insiders_set=selectInsider(SIFT_Des[0],num1,num2)
    for imgIdx in range(1,len(SIFT_Des)):
        print("Imge Idx : ",end='')
        print(imgIdx)
        Insiders=selectInsider(SIFT_Des[imgIdx],num1,num2)
        Insiders_set=np.concatenate((Insiders_set,Insiders),axis=0)
    return Insiders_set

def selectInsider(SIFT_Des_Img,num1=10,num2=100):    #Img 번째의 Image Descriptor중 , num1개의 클러스터로부터 가장 가까운 num2개만 반환합니다.
    center_points=small_K_means(SIFT_Des_Img,num1)
    dist=np.zeros(shape=len(SIFT_Des_Img),dtype=np.float)
    for desIdx in range(len(SIFT_Des_Img)):
        for centers in range(num1):
            dist[desIdx]+=L2Dist(center_points[centers],SIFT_Des_Img[desIdx])
    Insiders=np.argsort(dist)
    return SIFT_Des_Img[Insiders[:num2]]

def selectSpecimen(SIFT_Des,num):   #enter how many Des gonna be picked
    total_Des=SIFT_Des[0][np.asarray(len(SIFT_Des[0]) * np.random.rand(num), dtype=np.int)]
    for imgIdx in range(1,len(SIFT_Des)):
        selected_Des=SIFT_Des[imgIdx][np.asarray(len(SIFT_Des[imgIdx]) * np.random.rand(num), dtype=np.int)]
        total_Des=np.concatenate((total_Des,selected_Des),axis=0)
    return  total_Des


if __name__ =="__main__":
    main()
