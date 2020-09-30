import numpy as np
import cv2
import math
#sobel filter
Sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
def show_write(img,window_name,file_name,normalize=True):
    if normalize:
        newImg=normalize_Img(img)
    else:
        newImg=np.copy(img)
    cv2.imshow(window_name,newImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(file_name,newImg)
    del newImg
    return
def cross_correlation_1d(img, kernel):  # let kernerl size is over 1
    #    커널의 shape보고 vertical 인지 horizontal인지 판단.
    k = kernel.shape[0]  # kernel size
    padSize = int((k - 1) / 2)
    size = img.shape  # imput img size
    retImg = np.zeros(size,np.float_)
    if len(kernel.shape) == 1:  # horizontal kernel
        padded_img = np.zeros((size[0], size[1] + (k - 1)))
        padded_img[:, :padSize] = img[:, 0].reshape(size[0], 1)
        padded_img[:, padSize:padSize + size[1]] = img
        padded_img[:, padSize + size[1]:] = img[:, size[1] - 1].reshape(size[0], 1)
        temp1 = np.zeros(k, np.float_)
        for i in range(size[0]):
            for j in range(size[1]):
                temp1 = np.copy(padded_img[i, j:j + k])
                retImg[i,j]=np.sum(np.multiply(temp1, kernel))
        del temp1
    else:  # vertical kernel
        kernelT=kernel.reshape(1,k)
        padded_img = np.zeros((size[0]+(k-1), size[1]))
        padded_img[:padSize] = img[0]
        padded_img[padSize:padSize + size[0]] = img
        padded_img[padSize + size[0]:] = img[size[0] - 1]
        temp1 = np.zeros((k,1), np.float_)
        for i in range(size[0]):
            for j in range(size[1]):
                temp1 = np.copy(padded_img[i:i+k, j])
                retImg[i,j]=np.sum(np.multiply(temp1, kernelT))
        del temp1
    del padded_img
    return retImg
def cross_correlation_2d(img, kernel):  #overflow 조치!
    k = int(kernel.shape[0])  # kernel size
    k_half = int((k-1)/2)
    padSize = k_half
    size = img.shape  # input img size
    retImg = np.zeros(size,np.float_)
    padded_img=np.zeros((size[0]+2*padSize,size[1]+2*padSize),np.uint8)
    #fill padding part of 4 corners
    padded_img[:padSize,:padSize]=img[0][0]
    padded_img[:padSize,size[1]+padSize:]=img[0][size[1]-1]
    padded_img[size[0]+padSize:,:padSize]=img[size[0]-1][0]
    padded_img[size[0]+padSize:,size[1]+padSize:]=img[size[0]-1][size[1]-1]
    # expand the nearest pixel to 4 edges
    padded_img[padSize:size[0]+padSize, :padSize] = img[:, 0].reshape(size[0], 1)
    padded_img[padSize:size[0]+padSize, padSize + size[1]:] = img[:, size[1] - 1].reshape(size[0], 1)
    padded_img[:padSize,padSize:padSize+size[1]] = img[0]
    padded_img[padSize + size[0]:,padSize:padSize+size[1]] = img[size[0] - 1]
    #copy the origin image
    padded_img[padSize:size[0]+padSize,padSize:size[1]+padSize]=img
    temp1=np.zeros((k,k),np.float_)
    for i in range(size[0]):
        for j in range(size[1]):
            temp1 = np.copy(padded_img[i:i+k, j:j + k])
            temp1 = np.multiply(temp1,kernel)
            retImg[i,j]= np.sum(temp1)
    del temp1
    del padded_img
    return retImg
def get_gaussian_filter_1d(size, sigma):  # shape가 (size,)인 커널 만들어줌. vertival은 transpose해서 적용
    kernel=np.zeros(size,np.float_)
    rate = 1/(2*sigma*sigma)
    for i in range(1,int((size+1)/2)):
        kernel[int((size-1)/2-i)]=\
        kernel[int((size-1)/2+i)]=i*i
    kernel=np.exp(kernel*rate*(-1))*rate/math.pi
    kernel=kernel/np.sum(kernel)
    return kernel
def get_gaussian_filter_2d(size, sigma):
    kernel=np.zeros((size,size),np.float_)
    rate = 1/(2*sigma*sigma)
    xCords=np.zeros((size,size),np.int_)
    yCords=np.zeros((size,size),np.int_)
    for i in range(1,int((size-1)/2)+1):
        xCords[:,int((size-1)/2)-i]=xCords[:,int((size-1)/2)+i]=i
        yCords[int((size-1)/2)-i]=yCords[int((size-1)/2)+i]=i
    kernel=np.square(xCords)+np.square(yCords)
    kernel=np.exp(-1*kernel*rate)*(rate)/math.pi
    kernel=kernel/np.sum(kernel)
    return kernel
def compute_image_gradient(img):
    filtered_img=cross_correlation_2d(img,get_gaussian_filter_2d(7,1.5))
    dX= cross_correlation_2d(filtered_img, Sx)
    dY= cross_correlation_2d(filtered_img, Sy)
    temp1=np.array(dX,np.float_)
    temp2=np.array(dY,np.float_)
    mag=np.sqrt(np.square(temp1)+np.square(temp2))
    dir=np.arctan2(dY,dX)
    del temp1, temp2
    return mag, dir

def print_Result_by_Kernel(img, kernelSize,sigmaValues):
    ret=printResultBySigma(img,kernelSize[0],sigmaValues)
    for i in range(1,len(kernelSize)):
        newOne=printResultBySigma(img,kernelSize[i],sigmaValues)
        ret=np.concatenate((ret,newOne),axis=0)
        del newOne
    return ret

def printResultBySigma(img,kernelSize,sigmaValues):
    if len(kernelSize.shape)==0:    #kernel이 2D인 경우.
        label=str(kernelSize)+'X'+str(kernelSize)+' s='
        ret=cross_correlation_2d(img,get_gaussian_filter_2d(kernelSize,sigmaValues[0]))
        ret=np.array(ret,np.uint8)
        cv2.putText(ret,label+str(sigmaValues[0]), (10, 30), 2, 1, (0,0,0), 2)
        for i in range(1,len(sigmaValues)):
            newOne = cross_correlation_2d(img,get_gaussian_filter_2d(kernelSize,sigmaValues[i]))
            newOne = np.array(newOne,np.uint8)
            cv2.putText(newOne, label+str(sigmaValues[i]), (10, 30), 2, 1, (0, 0, 0), 2)
            ret=np.concatenate((ret,newOne),axis=1)
            del newOne
        return ret
    elif kernelSize[1]==1:       #vertical kernel
        label=str(kernelSize[0])+'X'+str(kernelSize[1])+' s='
        ret=cross_correlation_1d(img,get_gaussian_filter_1d(kernelSize[0],sigmaValues[0]).reshape(kernelSize[0],1))
        ret=np.array(ret,np.uint8)
        cv2.putText(ret,label+str(sigmaValues[0]), (10, 30), 2, 1, (0,0,0), 2)
        for i in range(1,len(sigmaValues)):
            newOne = cross_correlation_1d(img, get_gaussian_filter_1d(kernelSize[0], sigmaValues[i]).reshape(kernelSize[0], 1))
            newOne = np.array(newOne,np.uint8)
            cv2.putText(newOne, label+str(sigmaValues[i]), (10, 30), 2, 1, (0, 0, 0), 2)
            ret=np.concatenate((ret,newOne),axis=1)
            del newOne
        return ret
    else:                          #horizontal kernel
        label=str(kernelSize[0])+'X'+str(kernelSize[1])+' s='
        ret=cross_correlation_1d(img,get_gaussian_filter_1d(kernelSize[1],sigmaValues[0]))
        ret=np.array(ret,np.uint8)
        cv2.putText(ret,label+str(sigmaValues[0]), (10, 30), 2, 1, (0,0,0), 2)
        for i in range(1,len(sigmaValues)):
            newOne = cross_correlation_1d(img, get_gaussian_filter_1d(kernelSize[1], sigmaValues[i]))
            newOne = np.array(newOne,np.uint8)
            cv2.putText(newOne, label+str(sigmaValues[i]), (10, 30), 2, 1, (0, 0, 0), 2)
            ret=np.concatenate((ret,newOne),axis=1)
            del newOne
        return ret
def normalize_Img(img):
    retImg = np.absolute(img)
    retImg = retImg / np.max(retImg) * 255
    retImg=np.array(retImg,np.uint8)
    return retImg

    #NMS구현하기
def non_maximum_suppression_dir(mag, dir):
    sizes=np.shape(mag)
    retMag=np.copy(mag)
    for i in range(1,sizes[0]-1):
        for j in range(1,sizes[1]-1):
            if dir[i,j]>=math.pi/8*15 or dir[i,j]<math.pi/8:
                if mag[i,j]<mag[i,j+1] or mag[i,j]<mag[i,j-1]:
                    retMag[i,j]=0
            elif dir[i,j]>=math.pi/8*7 and dir[i,j]<math.pi/8*9:
                if mag[i,j]<mag[i,j+1] or mag[i,j]<mag[i,j-1]:
                    retMag[i,j]=0
            elif dir[i,j]>=math.pi/8*1 and dir[i,j]<math.pi/8*3:
                if mag[i,j]<mag[i+1,j+1] or mag[i,j]<mag[i-1,j-1]:
                    retMag[i,j]=0
            elif dir[i,j]>=math.pi/8*9 and dir[i,j]<math.pi/8*11:
                if mag[i,j]<mag[i+1,j+1] or mag[i,j]<mag[i-1,j-1]:
                    retMag[i,j]=0
            elif dir[i,j]>=math.pi/8*3 and dir[i,j]<math.pi/8*5:
                if mag[i,j]<mag[i+1,j] or mag[i,j]<mag[i-1,j]:
                    retMag[i,j]=0
            elif dir[i,j]>=math.pi/8*11 and dir[i,j]<math.pi/8*13:
                if mag[i,j]<mag[i+1,j] or mag[i,j]<mag[i-1,j]:
                    retMag[i,j]=0
            else:
                if mag[i,j]<mag[i+1,j-1] or mag[i,j]<mag[i-1,j+1]:
                    retMag[i,j]=0
    return retMag

#잘 안됐던 버전입니다.
def non_maximum_suppression_dir2(mag, dir):
    TF=retTF(mag,dir)       #bool 값이 False인 곳만 0 assign
    retMag=np.copy(mag)
    retMag[~TF]=0
    return retMag
def returnClosestLine(dir):
    dir2=np.array(dir,np.int_)
    dir2[np.logical_or(np.logical_or(dir>=math.pi/8*15, dir<math.pi/8),np.logical_and(dir>=math.pi/8*7, dir<math.pi/8*9))]=0
    dir2[np.logical_or(np.logical_and(dir>=math.pi/8*1, dir<math.pi/8*3),np.logical_and(dir>=math.pi/8*9, dir<math.pi/8*11))]=3
    dir2[np.logical_or(np.logical_and(dir>=math.pi/8*3, dir<math.pi/8*5),np.logical_and(dir>=math.pi/8*11, dir<math.pi/8*13))]=2
    dir2[np.logical_or(np.logical_and(dir>=math.pi/8*5, dir<math.pi/8*7),np.logical_and(dir>=math.pi/8*13, dir<math.pi/8*15))]=1
    # del dir
    return dir2

def retTF(mag,dir):
    sizes=np.shape(mag)
    ret = np.full_like(mag,dtype=bool,fill_value=True)
    pos=np.argwhere(ret).reshape(sizes[0],sizes[1],2)
    pos1, pos2 = ret2Pos(dir,pos)

    # #Filing values of 4 corners of ret numpy instance
    # boolV=True
    # if(dir[0,0]==1):
    #     if mag[0,0] < mag[0,1]:
    #         boolV=False
    # elif(dir[0,0]==2):
    #     pass
    # elif(dir[0,0]==3):
    #     if mag[0,0] < mag[1,0]:
    #         boolV=False
    # else:
    #     if mag[0,0] < mag[1,1]:
    #         boolV=False
    # ret[0,0]=boolV
    #
    # boolV=True
    # if(dir[0,sizes[1]-1]==1):
    #     if mag[0,sizes[1]-1] < mag[0,sizes[1]-2]:
    #         boolV=False
    # elif(dir[0,sizes[1]-1]==2):
    #     if mag[0,sizes[1]-1] < mag[1,sizes[1]-2]:
    #         boolV=False
    # elif(dir[0,sizes[1]-1]==3):
    #     if mag[0,sizes[1]-1] < mag[1,sizes[1]-1]:
    #         boolV=False
    # else:
    #     pass
    # ret[0,0]=boolV
    #
    # boolV=True
    # if(dir[sizes[0]-1,0]==1):
    #     if mag[sizes[0]-1,0] < mag[sizes[0]-1,1]:
    #         boolV=False
    # elif(dir[sizes[0]-1,0]==2):
    #     if mag[sizes[0]-1,0] < mag[sizes[0]-2,1]:
    #         boolV=False
    # elif(dir[sizes[0]-1,0]==3):
    #     if mag[sizes[0]-1,0] < mag[sizes[0] - 2, 0]:
    #         boolV = False
    # else:
    #     pass
    # ret[0,0]=boolV
    #
    # boolV=True
    # if(dir[sizes[0]-1,sizes[1]-1]==1):
    #     if mag[sizes[0]-1,sizes[1]-1] < mag[sizes[0]-1,sizes[1]-2]:
    #         boolV=False
    # elif(dir[sizes[0] - 1, sizes[1] - 1] == 2):
    #     pass
    # elif(dir[sizes[0] - 1, sizes[1] - 1] == 3):
    #     if mag[sizes[0]-1,sizes[1]-1] < mag[sizes[0]-2,sizes[1]-1]:
    #         boolV=False
    # else:
    #     if mag[sizes[0]-1,sizes[1]-1] < mag[sizes[0]-2,sizes[1]-2]:
    #         boolV=False
    # ret[0,0]=boolV
    #
    # #FIlling values of 4 edges of ret numpy instance
    # # edge1
    # boolNp=pos1[0,1:sizes[1]-1,0]==-1
    # posNp1=pos2[0,1:sizes[1]-1][boolNp]
    # posNp2=pos[0,1:sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[0,1:sizes[1]-1][boolNp]=TNp
    #
    # boolNp=pos2[0,1:sizes[1]-1,0]==-1
    # posNp1=pos1[0,1:sizes[1]-1][boolNp]
    # posNp2=pos[0,1:sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[0,1:sizes[1]-1][boolNp]=TNp
    # #edge2
    # boolNp=pos1[sizes[0]-1,1:sizes[1]-1,0]==sizes[0]
    # posNp1=pos2[sizes[0]-1,1:sizes[1]-1][boolNp]
    # posNp2=pos[sizes[0]-1,1:sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[sizes[0]-1,1:sizes[1]-1][boolNp]=TNp
    #
    # boolNp=pos2[sizes[0]-1,1:sizes[1]-1,0]==sizes[0]
    # posNp1=pos1[sizes[0]-1,1:sizes[1]-1][boolNp]
    # posNp2=pos[sizes[0]-1,1:sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[sizes[0]-1,1:sizes[1]-1][boolNp]=TNp
    # #edge3
    # boolNp=pos1[1:sizes[0]-1,0,1]==-1
    # posNp1=pos2[1:sizes[0]-1,0][boolNp]
    # posNp2=pos[1:sizes[0]-1,0][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[1:sizes[0]-1,0][boolNp]=TNp
    #
    # boolNp=pos2[1:sizes[0]-1,0,1]==-1
    # posNp1=pos1[1:sizes[0]-1,0][boolNp]
    # posNp2=pos[1:sizes[0]-1,0][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[1:sizes[0]-1,0][boolNp]=TNp
    # #edge4
    # boolNp=pos1[1:sizes[0]-1,sizes[1]-1,1]==sizes[1]
    # posNp1=pos2[1:sizes[0]-1,sizes[1]-1][boolNp]
    # posNp2=pos[1:sizes[0]-1,sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[1:sizes[0]-1,sizes[1]-1][boolNp]=TNp
    #
    # boolNp=pos2[1:sizes[0]-1,sizes[1]-1,1]==sizes[1]
    # posNp1=pos1[1:sizes[0]-1,sizes[1]-1][boolNp]
    # posNp2=pos[1:sizes[0]-1,sizes[1]-1][boolNp]
    # TNp=mag[posNp1[:,0],posNp1[:,1]]<mag[posNp2[:,0],posNp2[:,1]]
    # ret[1:sizes[0]-1,sizes[1]-1][boolNp]=TNp
    #
    #filling the inside part of ret numpy instance
    inside=np.logical_and(mag[pos[1:sizes[0]-1,1:sizes[1]-1,0],pos[1:sizes[0]-1,1:sizes[1]-1,1]]>=mag[pos1[1:sizes[0]-1,1:sizes[1]-1,0],pos1[1:sizes[0]-1,1:sizes[1]-1,1]],
                         mag[pos[1:sizes[0]-1,1:sizes[1]-1,0],pos[1:sizes[0]-1,1:sizes[1]-1,1]]>=mag[pos2[1:sizes[0]-1,1:sizes[1]-1,0],pos2[1:sizes[0]-1,1:sizes[1]-1,1]])
    ret[1:sizes[0]-1,1:sizes[1]-1]=inside
    del pos, pos1, pos2
    # del boolNp, posNp1, posNp2, TNp
    return ret
def ret2Pos(dir,pos): #dir에는 mapping 한 dir이 있음. pos에는 각 element의 위치가 저장되어 있음.
    dir=returnClosestLine(dir)
    pos1 = np.array(pos)
    pos2 = np.array(pos)
    dir0=(dir==0)
    dir1=(dir==1)
    dir2=(dir==2)
    dir3=(dir==3)
    #dir 0
    pos1[dir0,1]+=1
    pos2[dir0,1]-=1
    #dir 1
    pos1[dir1,0]-=1
    pos1[dir1,1]+=1
    pos2[dir1,0]+=1
    pos2[dir1,1]-=1
    #dir 2
    pos1[dir2,0]-=1
    pos2[dir2,0]+=1
    #dir 3
    pos1[dir3,0]+=1
    pos1[dir3,1]+=1
    pos2[dir3,0]-=1
    pos2[dir3,1]-=1
    return pos1, pos2

def compute_corner_response(img):
    filtered_img=cross_correlation_2d(img,get_gaussian_filter_2d(7,1.5))
    dX= cross_correlation_2d(filtered_img, Sx)
    dY= cross_correlation_2d(filtered_img, Sy)
    sizes=img.shape
    M=np.zeros(shape=(sizes[0],sizes[1],2,2))
    for i in range(2,sizes[0]-2):
        for j in range(2,sizes[1]-2):
            p_dX=dX[i-2:i+3,j-2:j+3]
            p_dY=dY[i-2:i+3,j-2:j+3]
            M[i,j,0,0]=np.sum(np.square(p_dX))
            M[i,j,1,1]=np.sum(np.square(p_dY))
            M[i,j,0,1]=M[i,j,1,0]=np.sum(np.multiply(p_dX,p_dY))
    R=np.zeros(shape=sizes)
    trace=np.empty(shape=sizes)
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            trace[i,j]=np.trace(M[i,j])
    trace=np.square(trace)
    R=np.linalg.det(M[:,:])-0.04*trace
    R[R<0]=0
    R=R/np.max(R)
    return R
def non_maximum_suppression_win(R,winSize=11):
    halfW=int((winSize-1)/2)
    sizes=R.shape
    ret=np.full(sizes,False)
    for i in range(halfW,sizes[0]-halfW-1):
        for j in range(halfW,sizes[1]-halfW-1):
            window=R[i-halfW:i+halfW+1,j-halfW:j+halfW+1]
            pos=np.argwhere(window==np.max(window))
            pos[:,0]+=i-halfW
            pos[:,1]+=j-halfW
            pos_i=pos[:,0]==i   #i-half를 빼고 비교해줘야 함.
            pos_j=pos[:,1]==j
            if(np.any(np.logical_and(pos_i,pos_j)) and R[i,j]>=0.1):
                ret[i,j]=True
    return ret