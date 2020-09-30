import numpy as np
import cv2
import sys
import time
import math
from scipy.spatial import distance
import pickle
import A2_functions as fn
from tqdm  import tqdm
from A3_a_functions import *
from compute_avg_reproj_error import *
import keyboard as kb
SET1_IMG1_PATH = "house1.jpg"
SET1_IMG2_PATH = "house2.jpg"
SET1_MATCH_PATH ="house_matches.txt"
SET2_IMG1_PATH = "library1.jpg"
SET2_IMG2_PATH = "library2.jpg"
SET2_MATCH_PATH ="library_matches.txt"
SET3_IMG1_PATH="temple1.png"
SET3_IMG2_PATH="temple2.png"
SET3_MATCH_PATH = "temple_matches.txt"
Matches_NP=[SET1_MATCH_PATH,SET2_MATCH_PATH,SET3_MATCH_PATH]
IMG1_NP=[SET1_IMG1_PATH,SET2_IMG1_PATH,SET3_IMG1_PATH]
IMG2_NP=[SET1_IMG2_PATH,SET2_IMG2_PATH,SET3_IMG2_PATH]
COLOR_RED=(0, 0, 255)
COLOR_GREEN=(0, 255, 0)
COLOR_BLUE= (255, 0, 0)
HYPER_PAR=[205,872,547]
def part1_mine(idx=0):
    M=np.loadtxt(Matches_NP[idx])
    srcP=M[:,:2]
    destP=M[:,2:]
    F=compute_F_mine(srcP,destP)
    error=compute_avg_reproj_error(M,F)
    return error

def part1_raw(idx=0): #set index is given
    M=np.loadtxt(Matches_NP[idx])
    srcP=M[:,:2]
    destP=M[:,2:]
    F=compute_F_raw(srcP,destP)
    error=compute_avg_reproj_error(M,F)
    return error

def part1_norm2(idx=0): #set index is given
    M=np.loadtxt(Matches_NP[idx])
    srcP=M[:,:2]
    destP=M[:,2:]
    F=compute_F_norm(srcP,destP)
    U, s, V = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F = np.dot(U, np.dot(np.diag(s), V))
    F/= F[2, 2]
    error=compute_avg_reproj_error(M,F)
    return error


def part1_norm(idx=0):  # set index is given
    M = np.loadtxt(Matches_NP[idx])
    srcP = M[:, :2]
    destP = M[:, 2:]
    F = compute_F_norm(srcP, destP)
    error = compute_avg_reproj_error(M, F)
    return error


def visualize_norm(idx):
    M=np.loadtxt(Matches_NP[idx])
    srcP=M[:,:2]
    destP=M[:,2:]
    F=compute_F_norm(srcP,destP)
    #Make F's Rank 2
    U, s, V = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F = np.dot(U, np.dot(np.diag(s), V))
    F/= F[2, 2]
    lines = np.zeros((len(M), 6), np.float)
    #Compute Epipole line Coefficient
    for i in range(len(M)):
        match=np.array([
            [srcP[i][0],destP[i][0]],
            [srcP[i][1],destP[i][1]],
            [1, 1],
        ])
        line_temp=np.dot(F,match)
        lines[i,0:3]=line_temp[:,0].T
        lines[i,3:6]=line_temp[:,1].T
    #Upload Images
    img1 = cv2.imread(IMG1_NP[idx], cv2.IMREAD_COLOR)
    img2 = cv2.imread(IMG2_NP[idx], cv2.IMREAD_COLOR)
    img_shape = img1.shape

    #Draw Epipole Lines
    rand_lines = np.zeros(shape=(3,6),dtype=np.float)

    lined_img = cv2.hconcat([img1, img2])
    cv2.imshow('Enter any key except q', lined_img)
    input_c=chr(cv2.waitKey())
    cv2.destroyAllWindows()
    while 1:
        if kb.is_pressed('q'):
            cv2.destroyAllWindows()
            return
        #select randomly 3 Matches
        rand_idx=select_3Points_Idx(len(M))
        #load their lines. order of RGB
        rand_lines=lines[rand_idx]
        start_end_points=np.zeros(shape=(6,4),dtype=np.int)
        for i in range(3):
            start_end_points[i*2:i*2+2]=ret_points(img_shape,rand_lines[i])
        #put lines on the coppied images
        lined_img1 = img1.copy()
        lined_img2 = img2.copy()
        cv2.line(lined_img1, tuple(start_end_points[0,0:2]), tuple(start_end_points[0,2:4]), COLOR_RED, 1)
        cv2.line(lined_img2, tuple(start_end_points[1,0:2]), tuple(start_end_points[1,2:4]), COLOR_RED, 1)
        cv2.line(lined_img1, tuple(start_end_points[2,0:2]), tuple(start_end_points[2,2:4]), COLOR_GREEN, 1)
        cv2.line(lined_img2, tuple(start_end_points[3,0:2]), tuple(start_end_points[3,2:4]), COLOR_GREEN, 1)
        cv2.line(lined_img1, tuple(start_end_points[4,0:2]), tuple(start_end_points[4,2:4]), COLOR_BLUE, 1)
        cv2.line(lined_img2, tuple(start_end_points[5,0:2]), tuple(start_end_points[5,2:4]), COLOR_BLUE, 1)

        cv2.circle(lined_img1, (int(srcP[rand_idx[0]][0]),int(srcP[rand_idx[0]][1])), 1, COLOR_RED, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[0]][0]),int(destP[rand_idx[0]][1])), 1, COLOR_RED, 5)
        cv2.circle(lined_img1, (int(srcP[rand_idx[1]][0]),int(srcP[rand_idx[1]][1])), 1, COLOR_GREEN, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[1]][0]),int(destP[rand_idx[1]][1])), 1, COLOR_GREEN, 5)
        cv2.circle(lined_img1, (int(srcP[rand_idx[2]][0]),int(srcP[rand_idx[2]][1])), 1, COLOR_BLUE, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[2]][0]),int(destP[rand_idx[2]][1])), 1, COLOR_BLUE, 5)

        lined_img = cv2.hconcat([lined_img1, lined_img2])

        cv2.imshow('Image with Epipolar lines', lined_img)
        input_c = chr(cv2.waitKey())
def visualize_mine(idx):
    M=np.loadtxt(Matches_NP[idx])
    srcP=M[:,:2]
    destP=M[:,2:]
    F=compute_F_mine(srcP,destP)
    #Make F's Rank 2
    U, s, V = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F = np.dot(U, np.dot(np.diag(s), V))
    F/= F[2, 2]
    lines = np.zeros((len(M), 6), np.float)
    #Compute Epipole line Coefficient
    for i in range(len(M)):
        match=np.array([
            [srcP[i][0],destP[i][0]],
            [srcP[i][1],destP[i][1]],
            [1, 1],
        ])
        line_temp=np.dot(F,match)
        lines[i,0:3]=line_temp[:,0].T
        lines[i,3:6]=line_temp[:,1].T
    #Upload Images
    img1 = cv2.imread(IMG1_NP[idx], cv2.IMREAD_COLOR)
    img2 = cv2.imread(IMG2_NP[idx], cv2.IMREAD_COLOR)
    img_shape = img1.shape

    #Draw Epipole Lines
    rand_lines = np.zeros(shape=(3,6),dtype=np.float)

    lined_img = cv2.hconcat([img1, img2])
    cv2.imshow('Enter any key except q', lined_img)
    input_c=chr(cv2.waitKey())
    cv2.destroyAllWindows()
    while 1:
        if kb.is_pressed('q'):
            cv2.destroyAllWindows()
            return
        #select randomly 3 Matches
        rand_idx=select_3Points_Idx(len(M))
        #load their lines. order of RGB
        rand_lines=lines[rand_idx]
        start_end_points=np.zeros(shape=(6,4),dtype=np.int)
        for i in range(3):
            start_end_points[i*2:i*2+2]=ret_points(img_shape,rand_lines[i])
        #put lines on the coppied images
        lined_img1 = img1.copy()
        lined_img2 = img2.copy()
        cv2.line(lined_img1, tuple(start_end_points[0,0:2]), tuple(start_end_points[0,2:4]), COLOR_RED, 1)
        cv2.line(lined_img2, tuple(start_end_points[1,0:2]), tuple(start_end_points[1,2:4]), COLOR_RED, 1)
        cv2.line(lined_img1, tuple(start_end_points[2,0:2]), tuple(start_end_points[2,2:4]), COLOR_GREEN, 1)
        cv2.line(lined_img2, tuple(start_end_points[3,0:2]), tuple(start_end_points[3,2:4]), COLOR_GREEN, 1)
        cv2.line(lined_img1, tuple(start_end_points[4,0:2]), tuple(start_end_points[4,2:4]), COLOR_BLUE, 1)
        cv2.line(lined_img2, tuple(start_end_points[5,0:2]), tuple(start_end_points[5,2:4]), COLOR_BLUE, 1)

        cv2.circle(lined_img1, (int(srcP[rand_idx[0]][0]),int(srcP[rand_idx[0]][1])), 1, COLOR_RED, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[0]][0]),int(destP[rand_idx[0]][1])), 1, COLOR_RED, 5)
        cv2.circle(lined_img1, (int(srcP[rand_idx[1]][0]),int(srcP[rand_idx[1]][1])), 1, COLOR_GREEN, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[1]][0]),int(destP[rand_idx[1]][1])), 1, COLOR_GREEN, 5)
        cv2.circle(lined_img1, (int(srcP[rand_idx[2]][0]),int(srcP[rand_idx[2]][1])), 1, COLOR_BLUE, 5)
        cv2.circle(lined_img2, (int(destP[rand_idx[2]][0]),int(destP[rand_idx[2]][1])), 1, COLOR_BLUE, 5)

        lined_img = cv2.hconcat([lined_img1, lined_img2])

        cv2.imshow('Image with Epipolar lines', lined_img)
        input_c = chr(cv2.waitKey())
def select_3Points_Idx(len_M):
    return np.asarray(len_M*np.random.rand(3),dtype=np.int)

def ret_points(img_shape,line):
    row=img_shape[0]
    col=img_shape[1]

    #img1 _ start_x start_y | end_x end_y
    #img2 _ start_x start_y | end_x end_y
    start_end_points=np.zeros(shape=(2,4),dtype=np.int)

    for i in range(2):
        row_cand1 = int(-line[2+i*3] / line[1+i*3])
        col_cand1 = int(-line[2+i*3] / line[0+i*3])
        row_cand2 = int((-line[2+i*3] - line[0+i*3] * col) / line[1+i*3])
        col_cand2 = int((-line[2+i*3] - line[1+i*3] * row) / line[0+i*3])
        idx=0
        if 0<= row_cand1 <=row:
            start_end_points[i][idx*2]=0
            start_end_points[i][idx*2+1]=row_cand1
            idx+=1
        if 0<= col_cand1<=col:
            start_end_points[i][idx*2]=col_cand1
            start_end_points[i][idx*2+1]=0
            idx+=1
        if 0<= row_cand2<=row:
            start_end_points[i][idx*2]=col
            start_end_points[i][idx*2+1]=row_cand2
            idx+=1
        if 0<= col_cand2<=col:
            start_end_points[i][idx*2]=col
            start_end_points[i][idx*2+1]=col_cand2
            idx+=1
    return start_end_points

def main():
    idxes=[0,1,2]
    for idx in idxes:
        print("Average Reprojection Errors ("+IMG1_NP[idx]+" and " +IMG2_NP[idx]+")")
        error0=part1_raw(idx)
        error1=part1_norm(idx)
        np.random.seed(HYPER_PAR[idx])
        error2 = part1_mine(idx)
        print("Raw = ",end='')
        print(error0)
        print("Norm = ",end='')
        print(error1)
        print("Mine = ",end='')
        print(error2)
        visualize_norm(idx)
        # visualize_norm(idx)
    pass


if __name__ == "__main__":
    main()