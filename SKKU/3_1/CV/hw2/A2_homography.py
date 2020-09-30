import numpy as np
import cv2
import sys
import time
import math
from scipy.spatial import distance
import pickle
import A2_functions as fn
from tqdm  import tqdm
import time
COVER_IMG_PATH = "cv_cover.jpg"
DESK_IMG_PATH = "cv_desk.png"
HP_IMG_PATH = "hp_cover.jpg"
HP_IMG_PATH = "hp_cover.jpg"
DIAMOND_HEAD_1="diamondhead-10.png"
DIAMOND_HEAD_2="diamondhead-11.png"
initial_T=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
matches_best=50 #이 갯수만큼 match를 sort함
# matches_num=500
diff_sum_th=325
def update_np_shortest_dist(np_shortest_dist):
    a=np.sort(np_shortest_dist[:,0],order='distance')
    np_shortest_dist_new=np.empty_like(np_shortest_dist)
    for i in range(len(np_shortest_dist)):
        for idx in range(len(np_shortest_dist)):
            if np_shortest_dist[idx,1]['fromKP']==a[i]['fromKP']:
                break
        np_shortest_dist_new[i]=np_shortest_dist[idx]
    del np_shortest_dist
    np_shortest_dist=np_shortest_dist_new
    return np_shortest_dist
def BFMatcher(des1,des2):
    matches_num=min(len(des1),len(des2))
    #calculate
    np_des1=np.empty(shape=matches_num,dtype=object)
    np_des2=np.empty(shape=matches_num,dtype=object)
    # for i in range(matches_num):
    print("Encoding des to binary codes")
    for i in tqdm(range(0, matches_num), mininterval=1):
        x=""
        y=""
        for j in range(32):
            x += np.binary_repr(des1[i][j], width=8)
            y += np.binary_repr(des2[i][j], width=8)
        np_des1[i]=x
        np_des2[i]=y
    np_hamm_dist=np.empty(shape=(matches_num,matches_num),dtype=float)
    print()
    print("Calculating all Hamming distances")
    for i in tqdm(range(0, matches_num), mininterval=1):
        for j in range(matches_num):
            np_hamm_dist[i][j]=distance.hamming(list(np_des1[i]),list(np_des2[j]))

    np_shortest_dist=np.empty(shape=(matches_num,matches_num),dtype=[('distance','f8'),('toKP','i8'),('fromKP','i8')])
    np_shortest_dist['distance']=np_hamm_dist
    for i in range(matches_num):
        np_shortest_dist[i]['fromKP'][:]=i
        np_shortest_dist[i]['toKP'][:]=range(matches_num)
    for i in range(matches_num):
        np_shortest_dist[i]=np.sort(np_shortest_dist[i],order='distance')

    np_best_match = np.empty(shape=matches_best, dtype=[('distance', 'f8'), ('toKP', 'i8'), ('fromKP', 'i8')])
    b=np.full(shape=matches_num+1,fill_value=False)
    np_shortest_dist=update_np_shortest_dist(np_shortest_dist)
    # for i in range(10):
    for i in tqdm(range(0, matches_best), mininterval=1):
    # for i in range(matches_best):
        while b[np_shortest_dist[0, 0]['toKP']] != False:
            np_shortest_dist[0, 0]['distance'] = 1
            np_shortest_dist[0]=np.sort(np_shortest_dist[0],order='distance')
            np_shortest_dist = update_np_shortest_dist(np_shortest_dist)
        np_best_match[i] = np_shortest_dist[0, 0]
        b[np_best_match[i]['toKP']] = True
        np_shortest_dist = np.delete(np_shortest_dist, 0, 0)
    return np_best_match
def normalize(points):
    ret_points=np.copy(points)
    T1=np.copy(initial_T)
    T2=np.copy(initial_T)
    T1[0,2]=-np.mean(ret_points,axis=0)[0]
    T1[1,2]=-np.mean(ret_points,axis=0)[1]
    maxNorm=np.max(np.linalg.norm(ret_points,axis=1))
    T2[0,0]=(1/maxNorm)*math.sqrt(2)
    T2[1,1]=(1/maxNorm)*math.sqrt(2)
    ret_points=ret_points-np.mean(ret_points,axis=0)
    ret_points/=maxNorm
    ret_points*=math.sqrt(2)
    T2=T2.dot(T1)
    del T1
    return T2, ret_points

def compute_homography ( srcP , destP ):
    T_s,norm_srcP=normalize(srcP)
    T_d,norm_destP=normalize(destP)
    H=np.empty(shape=(3,3),dtype=float)
    A=np.zeros(shape=(len(srcP)*2,9),dtype=float)
    for i in range(len(srcP)):
        A[i*2][:2]=norm_srcP[i]*-1
        A[i*2][2]=-1
        A[i * 2+1,3:5] = norm_srcP[i] * -1
        A[i * 2+1,5] = -1
        A[i * 2:i*2+2,6:8] = norm_srcP[i]
        A[i*2,6:8]*=norm_destP[i][0]
        A[i*2+1,6:8]*=norm_destP[i][1]
        A[i*2,8]=norm_destP[i][0]
        A[i * 2+1,8] = norm_destP[i][1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v=np.transpose(vh)
    H=v[:,8]
    try:
        H/=H[8]
    except:
        H/=0.00000000001
    H=np.reshape(H,(3,3))
    fromS_toP=np.linalg.inv(T_d).dot(H).dot(T_s)
    return fromS_toP

#new version
#해당 갯수만큼 들어왔다고 판단. matches_ransac_par갯수만큼만 본다
def compute_homography_ransac ( srcP , destP , th=1.4): #srcP & destP are not normalized
    matches_ransac_par = min(len(srcP),len(destP))
    T_s,norm_srcP=normalize(srcP)
    T_d,norm_destP=normalize(destP)

    #frist RANDOM group
    rand_selelct=np.random.choice(matches_ransac_par, 4)
    H=compute_homography(norm_srcP[rand_selelct], norm_destP[rand_selelct])
    fromS_toP=np.linalg.inv(T_d).dot(H).dot(T_s)
    final_destP=np.zeros(shape=(matches_ransac_par,2),dtype=np.float)
    for i in range(matches_ransac_par):
        final_destP[i]=fn.findLocation(fromS_toP,np.append(srcP[i],1)).reshape(2)
    count=0
    diff=destP-final_destP
    diff_sum = np.sum(np.linalg.norm(diff, axis=1))
    try:
        inliners=np.linalg.norm(diff, axis=1)<th
        score=np.sum(inliners)
    except:
        count=20
    pbar = tqdm(total=500)#1500
    # while count<19000 and score<matches_ransac_par-1:
    while(1):
        if(matches_ransac_par==25):
            if(score>18):
                break
        else:
            if(score==35):
                break
        if(count>10):
            rand_selelct = np.random.choice(matches_ransac_par, 4)
            H = compute_homography(norm_srcP[rand_selelct], norm_destP[rand_selelct])
            fromS_toP = np.linalg.inv(T_d).dot(H).dot(T_s)
            final_destP = np.zeros(shape=(matches_ransac_par, 2), dtype=np.float)
            for i in range(matches_ransac_par):
                final_destP[i] = fn.findLocation(fromS_toP, np.append(srcP[i], 1)).reshape(2)
            diff = destP - final_destP
            diff_sum = np.sum(np.linalg.norm(diff, axis=1))
            try:
                inliners = np.linalg.norm(diff, axis=1) < th
                score = np.sum(inliners)
                count=0
            except:
                count=20
        pbar.update(1)
        H = compute_homography(norm_srcP[inliners], norm_destP[inliners])
        fromS_toP = np.linalg.inv(T_d).dot(H).dot(T_s)
        final_destP = np.zeros(shape=(matches_ransac_par, 2), dtype=np.float)
        for i in range(matches_ransac_par):
            final_destP[i] = fn.findLocation(fromS_toP, np.append(srcP[i], 1)).reshape(2)
        diff = destP - final_destP
        diff_sum = np.sum(np.linalg.norm(diff, axis=1))
        try:
            inliners = np.linalg.norm(diff, axis=1) < th
            score = np.sum(inliners)
        except:
            count=20
        count+=1
    #
    #
    # while count<500 and score < matches_ransac_par:
    #     pbar.update(1)
    #     H = compute_homography(norm_srcP[inliners], norm_destP[inliners])
    #     fromS_toP = np.linalg.inv(T_d).dot(H).dot(T_s)
    #     final_destP = np.zeros(shape=(matches_ransac_par, 2), dtype=np.float)
    #     for i in range(matches_ransac_par):
    #         final_destP[i] = fn.findLocation(fromS_toP, np.append(srcP[i], 1)).reshape(2)
    #     diff = destP - final_destP
    #     diff_sum = np.sum(np.linalg.norm(diff, axis=1))
    #     inliners = np.linalg.norm(diff, axis=1) < th
    #     score = np.sum(inliners)
    # pbar.close()
    del norm_srcP, norm_destP
    return fromS_toP
def main():
    #input
    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
    img_cover=cv2.imread(COVER_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    img_desk=cv2.imread(DESK_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    hp_cover=cv2.imread(HP_IMG_PATH,cv2.IMREAD_GRAYSCALE)
    img_dia1=cv2.imread(DIAMOND_HEAD_1,cv2.IMREAD_GRAYSCALE)
    img_dia2=cv2.imread(DIAMOND_HEAD_2,cv2.IMREAD_GRAYSCALE)

    #2-1
    orb = cv2.ORB_create()
    kp_cover = orb.detect( img_cover , None )
    kp_cover, des_cover = orb.compute( img_cover , kp_cover)
    kp_desk = orb.detect( img_desk , None )
    kp_desk, des_desk = orb.compute( img_desk , kp_desk)

    np_best_match = BFMatcher(des_cover, des_desk)
    matches=[]
    #2-1
    for i in range(len(np_best_match)):
        matches.append(cv2.DMatch(np_best_match[i]['toKP'],np_best_match[i]['fromKP'],0))
    res=cv2.drawMatches(img_desk,kp_desk,img_cover,kp_cover,matches[:10],None,flags=2)
    cv2.imshow('Feature Matching',res)
    cv2.imwrite('2_1_Feature Matching.png',res)
    cv2.waitKey()
    cv2.destroyAllWindows()


    #2-2
    #srcP=cover / destP = desk
    matches_normal = 15
    srcP=np.empty(shape=(matches_normal,2),dtype=float)
    destP=np.empty(shape=(matches_normal,2),dtype=float)
    for i in range(matches_normal):
        srcP[i][0]=kp_cover[matches[i].trainIdx].pt[0]
        srcP[i][1]=kp_cover[matches[i].trainIdx].pt[1]
        destP[i][0]=kp_desk[matches[i].queryIdx].pt[0]
        destP[i][1]=kp_desk[matches[i].queryIdx].pt[1]

    #normalize and get transformation matrix
    H_normal=compute_homography(srcP, destP)
    # #모서리들 보내기
    # height, width = img_cover.shape
    dst_normal = cv2.warpPerspective(img_cover,H_normal,(img_desk.shape[1],img_desk.shape[0]))

    roi=np.copy(img_desk)
    img2gray = dst_normal
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(dst_normal, dst_normal, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst_normal = cv2.add(img1_fg, img2_bg)

    cv2.imshow("2_2_H_dst_normal", dst_normal)
    cv2.imwrite('2_2_H_normal.png',dst_normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #2-3
    #srcP=cover / destP = desk
    matches_ransac=25
    srcP=np.empty(shape=(matches_ransac,2),dtype=float)
    destP=np.empty(shape=(matches_ransac,2),dtype=float)
    for i in range(matches_ransac):
        srcP[i][0]=kp_cover[matches[i].trainIdx].pt[0]
        srcP[i][1]=kp_cover[matches[i].trainIdx].pt[1]
        destP[i][0]=kp_desk[matches[i].queryIdx].pt[0]
        destP[i][1]=kp_desk[matches[i].queryIdx].pt[1]
    H_ransac = compute_homography_ransac(srcP, destP,th=4)
    roi = np.copy(img_desk)
    dst_ransac = cv2.warpPerspective(img_cover, H_ransac, (img_desk.shape[1], img_desk.shape[0]))
    img2gray = dst_ransac
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(dst_ransac, dst_ransac, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst_ransac = cv2.add(img1_fg, img2_bg)
    cv2.imshow('2_3_H_ransac', dst_ransac)
    cv2.imwrite('2_3_H_ransac.png', dst_ransac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hp_cover = cv2.resize(hp_cover, (img_cover.shape[1],img_cover.shape[0]), interpolation=cv2.INTER_AREA)
    roi = np.copy(img_desk)
    dst_ransac = cv2.warpPerspective(hp_cover, H_ransac, (img_desk.shape[1], img_desk.shape[0]))
    img2gray = dst_ransac
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_fg = cv2.bitwise_and(dst_ransac, dst_ransac, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    dst_ransac = cv2.add(img1_fg, img2_bg)

    cv2.imshow('2_4_H_ransac', dst_ransac)
    cv2.imwrite('2_4_H_ransac.png', dst_ransac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #2-5
    orb = cv2.ORB_create()
    kp_dia1 = orb.detect(img_dia1 , None )
    kp_dia1, des_dia1 = orb.compute( img_dia1 , kp_dia1)
    kp_dia2 = orb.detect( img_dia2, None )
    kp_dia2, des_dia2 = orb.compute( img_dia2, kp_dia2)
    np_best_match_dia = BFMatcher(des_dia2, des_dia1)
    matches=[]
    for i in range(len(np_best_match_dia)):
        matches.append(cv2.DMatch(np_best_match_dia[i]['toKP'],np_best_match_dia[i]['fromKP'],0))

    matches_ransac = 35
    srcP = np.empty(shape=(matches_ransac, 2), dtype=float)
    destP = np.empty(shape=(matches_ransac, 2), dtype=float)
    for i in range(matches_ransac):
        srcP[i][0] = kp_dia2[matches[i].trainIdx].pt[0]
        srcP[i][1] = kp_dia2[matches[i].trainIdx].pt[1]
        destP[i][0] = kp_dia1[matches[i].queryIdx].pt[0]
        destP[i][1] = kp_dia1[matches[i].queryIdx].pt[1]

    H_ransac= compute_homography_ransac(srcP, destP, th=4)
    img_dia1_bigger=np.copy(img_dia1)
    img_dia1_bigger=np.concatenate((img_dia1_bigger,np.zeros(shape=img_dia1.shape,dtype=np.uint8)),axis=1)
    img_dia1_bigger=np.concatenate((img_dia1_bigger,np.zeros(shape=(img_dia1.shape[0],img_dia1.shape[1]*2),dtype=np.uint8)),axis=0)


    dst_ransac = cv2.warpPerspective(img_dia2, H_ransac, (img_dia1.shape[1]*2, img_dia2.shape[0]*2))
    B=np.copy(dst_ransac)
    roi = np.copy(dst_ransac)
    img2gray = img_dia1_bigger
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_fg = cv2.bitwise_and(img_dia1_bigger, img_dia1_bigger, mask=mask)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # #
    img_dia1_bigger = cv2.add(img1_fg, img2_bg)

    # final_ransac=np.copy(img_dia1_bigger[:img_dia1.shape[0],:img_dia1.shape[1]])
    dia_size=img_dia1.shape
    for i in range(dia_size[1]*2):
        if np.sum(img_dia1_bigger[dia_size[0]-1,i:i+15])==0:
            img_edge=i
            break
    dia_stitched=np.copy(img_dia1_bigger[:dia_size[0],:img_edge])
    cv2.imshow('2_5_a_H_ransac',dia_stitched)
    cv2.imwrite('2_5_a_H_ransac.png',dia_stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()


    length=100
    for i in range(length):
        rate=float(i/length)
        rate2=1-rate
        dia_stitched[:,dia_size[1]-100+i]=rate*img_dia1[:,dia_size[1]-100+i]+rate2*B[:dia_size[0],dia_size[1]-100+i]
    cv2.imshow('2_5_b_gradation',dia_stitched)
    cv2.imwrite('2_5_b_gradation.png',dia_stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()



    # cv2.imshow('2_5_a_Image stitching',img_dia1_bigger)
    # cv2.imwrite('2_5_a_Image stitching.png',img_dia1_bigger)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # final_ransac=np.copy(img_dia1_bigger[:img_dia1.shape[0],:img_dia1.shape[1]])
    # cv2.imshow('2_5 a_ Image stitching',final_ransac)
    # cv2.imwrite('2_5 a_ Image stitching.png',final_ransac)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
