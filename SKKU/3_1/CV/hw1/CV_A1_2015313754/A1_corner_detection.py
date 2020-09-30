import cv2
import numpy as np
import sys
import math
import time
import fnimp as fn
#import function implementations as fn
imgNames=['lenna.png','shapes.png']
def main():
    lenna_img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes_img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

    #lenna
    #corner detction
    startTime_corner_response_lenna = time.time()
    corners_lenna=fn.compute_corner_response(lenna_img)
    endtTime_corner_response_lenna = time.time()
    print("Time consumed for getting corners of lenna image is "+str(endtTime_corner_response_lenna-startTime_corner_response_lenna))
    fn.show_write(corners_lenna,'part_3_corner_ raw_corners','./result/part_3_corner_ raw_corners.png')
    #change color of pixel
    pos=corners_lenna>0.1
    corners_lenna_norm=fn.normalize_Img(corners_lenna)
    lenna_img_rgb = cv2.cvtColor(lenna_img, cv2.COLOR_GRAY2RGB)
    corners_lenna_bgr = cv2.cvtColor(corners_lenna_norm, cv2.COLOR_GRAY2RGB)
    b,g,r=cv2.split(corners_lenna_bgr)
    g[pos]=255
    inversebgr_lenna = cv2.merge((r, g, b))
    ret=cv2.add(lenna_img_rgb,inversebgr_lenna)
    fn.show_write(ret,'part_3_corner_bin_lenna','./result/part_3_corner_bin_lenna.png')

    #NMS
    startTime_corner_sup_lenna = time.time()
    filter=fn.non_maximum_suppression_win(corners_lenna,11)
    endtTime_corner_sup_lenna = time.time()
    #pointing circles
    position=np.argwhere(filter==True)
    corners_lenna[~filter]=0
    for point in position:
        cv2.circle(lenna_img_rgb,tuple((point[1],point[0])),4,(0,255,0),2)
    print("Time consumed for getting local maximums of R values of lenna image is "+str(endtTime_corner_sup_lenna-startTime_corner_sup_lenna))
    fn.show_write(lenna_img_rgb,'part_3_corner_sup_lenna','./result/part_3_corner_sup_lenna.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Shapes
    #corner detction
    startTime_corner_response_shapes = time.time()
    corners_shapes=fn.compute_corner_response(shapes_img)
    endtTime_corner_response_shapes = time.time()
    print("Time consumed for getting corners of shapes image is "+str(endtTime_corner_response_shapes-startTime_corner_response_shapes))
    fn.show_write(corners_shapes,'part_3_corner_ raw_corners','./result/part_3_corner_ raw_corners.png')
    #change color of pixel
    pos=corners_shapes>0.1
    corners_shapes_norm=fn.normalize_Img(corners_shapes)
    shapes_img_rgb = cv2.cvtColor(shapes_img, cv2.COLOR_GRAY2RGB)
    corners_shapes_bgr = cv2.cvtColor(corners_shapes_norm, cv2.COLOR_GRAY2RGB)
    b,g,r=cv2.split(corners_shapes_bgr)
    g[pos]=255
    inversebgr_shapes = cv2.merge((r, g, b))
    ret=cv2.add(shapes_img_rgb,inversebgr_shapes)
    fn.show_write(ret,'part_3_corner_bin_shapes','./result/part_3_corner_bin_shapes.png')

    #NMS
    startTime_corner_sup_shapes = time.time()
    filter=fn.non_maximum_suppression_win(corners_shapes,11)
    endtTime_corner_sup_shapes = time.time()
    #making circles
    position=np.argwhere(filter==True)
    corners_shapes[~filter]=0
    for point in position:
        cv2.circle(shapes_img_rgb,tuple((point[1],point[0])),4,(0,255,0),2)
    print("Time consumed for getting local maximums of R values of shapes image is "+str(endtTime_corner_sup_shapes-startTime_corner_sup_shapes))
    fn.show_write(shapes_img_rgb,'part_3_corner_sup_shapes','./result/part_3_corner_sup_shapes.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
if __name__ == "__main__":
    main()
