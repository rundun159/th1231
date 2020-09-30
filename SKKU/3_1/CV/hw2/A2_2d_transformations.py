import numpy as np
import cv2
import sys
import time
import math
IMAGE_FILE_PATH = "smile.png"
move_initial=np.array([[1,0,0],[0,1,0],[0,0,1]])
move_a=np.array([[1,0,0],[0,1,-5],[0,0,1]])
move_d=np.array([[1,0,0],[0,1,5],[0,0,1]])
move_w=np.array([[1,0,-5],[0,1,0],[0,0,1]])
move_s=np.array([[1,0,5],[0,1,0],[0,0,1]])

move_r=np.array([[math.cos(math.pi/36),-math.sin(math.pi/36),0],[math.sin(math.pi/36),math.cos(math.pi/36),0],[0,0,1]])
move_R=np.array([[math.cos(math.pi/36),math.sin(math.pi/36),0],[-math.sin(math.pi/36),math.cos(math.pi/36),0],[0,0,1]])

move_f=np.array([[1,0,0],[0,-1,0],[0,0,1]])
move_F=np.array([[-1,0,0],[0,1,0],[0,0,1]])

move_y=np.array([[0.95,0,0],[0,1,0],[0,0,1]])
move_Y=np.array([[1.05,0,0],[0,1,0],[0,0,1]])
move_x=np.array([[1,0,0],[0,0.95,0],[0,0,1]])
move_X=np.array([[1,0,0],[0,1.05,0],[0,0,1]])

def findLocation(M,x): #shape of M : 3x3 , shape of x : 1x3. shape of return vector : 1x2
    ret=M.dot(x.reshape(3,1))
    ret=ret/ret[2]
    return ret[:2]
def show_write(img,window_name,file_name,normalize=True):
    newImg=np.copy(img)
    cv2.imshow(window_name,newImg)
    cv2.imwrite(file_name,newImg)
    del newImg
    return
def get_transformed_image(img, M): #img에 M Transformation을 수행한 결과를 801x801 plane에 보여주기
    plane=np.full((801,801),255, np.uint8)
    par=np.array([0,0,1])
    shape_img=img.shape
    T_i = np.array([[1, 0, -shape_img[0]//2+1], [0, 1, -shape_img[1]//2+1], [0, 0, 1]])
    D = np.array([[1, 0, 400], [0, 1, 400], [0, 0, 1]])
    for i in range(shape_img[0]):
        par[0] = i
        for j in range(shape_img[1]):
            par[1]=j
            a=findLocation(M,np.append(findLocation(T_i,par),1).reshape(3,1))
            a=findLocation(D,np.append(a,1).reshape(3,1))
            plane[int(a[0]),int(a[1])]=img[i,j]
    return plane
# def use_cache(img,M):
#     par=np.array([0,0,1])
#     shape_img=img.shape
#     D = np.array([[1, 0, 400], [0, 1, 400], [0, 0, 1]])
#     for i in range(shape_img[0]):
#         par[0] = i
#         for j in range(shape_img[1]):
#             par[1]=j
#
def main():
    img=cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_GRAYSCALE)
    M=np.array([[1,0,0],[0,1,0],[0,0,1]])
    plane=get_transformed_image(img,M)
    cv2.arrowedLine(plane,(400,800),(400,0),0,thickness=2,tipLength=0.03)
    cv2.arrowedLine(plane,(0,400),(800,400),0,thickness=2,tipLength=0.03)
    # plane = bojeong(plane)
    cv2.imshow("smile",plane)
    input_c=chr(cv2.waitKey())
    while(input_c!='Q'):
        print(input_c)
        cv2.destroyAllWindows()
        if input_c=='a':
            M=move_a.dot(M)
        if input_c=='d':
            M=move_d.dot(M)
        if input_c=='w':
            M=move_w.dot(M)
        if input_c=='s':
            M=move_s.dot(M)
        if input_c=='r':
            M=move_r.dot(M)
        if input_c=='R':
            M=move_R.dot(M)
        if input_c=='f':
            M=move_f.dot(M)
        if input_c=='F':
            M=move_F.dot(M)
        if input_c=='x':
            M=move_x.dot(M)
        if input_c=='X':
            M=move_X.dot(M)
        if input_c=='y':
            M=move_y.dot(M)
        if input_c=='Y':
            M=move_Y.dot(M)
        if input_c=='H':
            M=move_initial
        plane=get_transformed_image(img,M)
        cv2.arrowedLine(plane, (400, 800), (400, 0), 0, thickness=2, tipLength=0.03)
        cv2.arrowedLine(plane, (0, 400), (800, 400), 0, thickness=2, tipLength=0.03)
        # plane = bojeong(plane)
        cv2.imshow("smile", plane)
        input_c = chr(cv2.waitKey())
    cv2.destroyAllWindows()
def bojeong(plane):
    shapes=plane.shape
    plane2=np.copy(plane)
    for i in range(10,shapes[0]-10):
        for j in range(10,shapes[1]-10):
            if plane[i-1,j]==0 and plane[i+1,j]==0:
                plane2[i,j]=0
            if plane[i,j+1]==0 and plane[i,j+1]==0:
                plane2[i,j]=0
            if plane[i+1,j+1]==0 and plane[i-1,j-1]==0:
                plane2[i,j]=0
            if plane[i-1,j+1]==0 and plane[i+1,j-1]==0:
                plane2[i,j]=0
    del plane
    return plane2
if __name__ == "__main__":
    main()