import cv2
import numpy as np
import sys
import time
import fnimp as fn
#import function implementations as fn
#cross_correlation_1d(img, kernel):
imgNames=['lenna.png','shapes.png']
def main():
    lenna_img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes_img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    print("result of get_gaussian_filter_1d(5,1)")
    print(fn.get_gaussian_filter_1d(5,1))
    print("result of get_gaussian_filter_2d(5,1)")
    print(fn.get_gaussian_filter_2d(5,1))
    kernelSizes1=np.array([5,11,17],np.int32)
    kernelSizes2=np.array([(5,1),(11,1),(17,1)])        #vertical kernel
    kernelSizes3=np.array([(1,5),(1,11),(1,17)])        #horizontal kernel
    sigmaValues=np.array([1,6,11])

    #lenna.png
    #9가지 필터를 적용한 결과 출력 & 파일 저장
    lenna_2D=fn.print_Result_by_Kernel(lenna_img,kernelSizes1,sigmaValues)
    fn.show_write(lenna_2D,'part_1_gaussian_filtered_lenna','./result/part_1_gaussian_filtered_lenna.png',False)

    #2D, 1D_Vertical, 1D_Horizontal 필터링 & 경과 시간 출력
    #2D는 11x11 s=6 적용
    startTime_2D = time.time()
    lenna_2D=fn.cross_correlation_2d(lenna_img,fn.get_gaussian_filter_2d(11,6))
    endTime_2D = time.time()
    print("Time consumed for filtering lenna image by 2D(11x11, s=6) filter is "+str(endTime_2D-startTime_2D))
    #1D는 11x1 s=6 적용
    startTime_1D_V = time.time()
    lenna_1D_V=fn.cross_correlation_1d(lenna_img,fn.get_gaussian_filter_1d(11,1).reshape(11,1))
    endTime_1D_V = time.time()
    print("Time consumed for filtering lenna image by 1D(11x1, s=6) Vertical filter is "+str(endTime_1D_V-startTime_1D_V))
    #1D는 1x11 s=6 적용
    startTime_1D_H = time.time()
    lenna_1D_H=fn.cross_correlation_1d(lenna_img,fn.get_gaussian_filter_1d(11,1))
    endTime_1D_H = time.time()
    print("Time consumed for filtering lenna image by 1D(1x111, s=6) Horizontal filter is "+str(endTime_1D_H-startTime_1D_H))

    #Difference map 구하기
    diff_1D_V=lenna_2D-lenna_1D_V
    diff_1D_H=lenna_2D-lenna_1D_H
    #차이값 출력하기
    print("The sum of intensity differences between filtered images by 2D and vertical 1D filters is "+str(np.sum(np.absolute(diff_1D_V))))
    print("The sum of intensity differences between filtered images by 2D and horizontal 1D filters is "+str(np.sum(np.absolute(diff_1D_H))))
    #Difference map 보여주기
    fn.show_write(fn.normalize_Img(diff_1D_V),'Difference map between filtered images by 2D and vertical 1D filters','./result/part_1_difference_map_with_1D_vertical_lenna.png')
    fn.show_write(fn.normalize_Img(diff_1D_H),'Difference map between filtered images by 2D and horizontal 1D filters','./result/part_1_difference_map_with_1D_horizontal_lenna.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #shapes.png
    shapes_2D=fn.print_Result_by_Kernel(shapes_img,kernelSizes1,sigmaValues)
    fn.show_write(shapes_2D,'part_1_gaussian_filtered_shapes','./result/part_1_gaussian_filtered_shapes.png',False)
    #2D, 1D_Vertical, 1D_Horizontal 필터링 & 경과 시간 출력
    #2D는 11x11 s=6 적용
    startTime_2D = time.time()
    shapes_2D=fn.cross_correlation_2d(shapes_img,fn.get_gaussian_filter_2d(11,6))
    endTime_2D = time.time()
    print("Time consumed for filtering shapes image by 2D(11x11, s=6) filter is "+str(endTime_2D-startTime_2D))
    #1D는 11x1 s=6 적용
    startTime_1D_V = time.time()
    shapes_1D_V=fn.cross_correlation_1d(shapes_img,fn.get_gaussian_filter_1d(11,1).reshape(11,1))
    endTime_1D_V = time.time()
    print("Time consumed for filtering shapes image by 1D(11x1, s=6) Vertical filter is "+str(endTime_1D_V-startTime_1D_V))
    #1D는 1x11 s=6 적용
    startTime_1D_H = time.time()
    shapes_1D_H=fn.cross_correlation_1d(shapes_img,fn.get_gaussian_filter_1d(11,1))
    endTime_1D_H = time.time()
    print("Time consumed for filtering shapes image by 1D(1x111, s=6) Horizontal filter is "+str(endTime_1D_H-startTime_1D_H))

    #Difference map 구하기
    diff_1D_V=shapes_2D-shapes_1D_V
    diff_1D_H=shapes_2D-shapes_1D_H
    #차이값 출력하기
    print("The sum of intensity differences between filtered images by 2D and vertical 1D filters is "+str(np.sum(np.absolute(diff_1D_V))))
    print("The sum of intensity differences between filtered images by 2D and horizontal 1D filters is "+str(np.sum(np.absolute(diff_1D_H))))
    #Difference map 보여주기
    fn.show_write(fn.normalize_Img(diff_1D_V),'Difference map between filtered images by 2D and vertical 1D filters','./result/part_1_difference_map_with_1D_vertical_shapes.png')
    fn.show_write(fn.normalize_Img(diff_1D_H),'Difference map between filtered images by 2D and horizontal 1D filters','./result/part_1_difference_map_with_1D_horizontal_shapes.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
if __name__ == "__main__":
    main()
