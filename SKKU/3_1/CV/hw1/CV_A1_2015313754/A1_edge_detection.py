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
    #Edge detection
    startTime_edge_raw_lenna = time.time()
    lenna_mag, lenna_dir=fn.compute_image_gradient(lenna_img)
    endTime_edge_raw_lenna = time.time()
    print("Time consumed for getting edge raw of lenna image is "+str(endTime_edge_raw_lenna-startTime_edge_raw_lenna))
    fn.show_write(fn.normalize_Img(lenna_mag),'part_2_edge_raw_ lenna','./result/part_2_edge_raw_lenna.png')

    #non maximum suprresion
    startTime_NMS_lenna = time.time()
    lenna_mag, lenna_dir=fn.compute_image_gradient(lenna_img)
    endTime_NMS_lenna = time.time()
    # NMS_lenna=fn.non_maximum_suppression_dir(lenna_mag,lenna_dir)
    NMS_lenna=fn.non_maximum_suppression_dir(lenna_mag,lenna_dir+math.pi)
    print("Time consumed for suppressing gradients of lenna image is "+str(endTime_NMS_lenna-startTime_NMS_lenna))
    fn.show_write(fn.normalize_Img(NMS_lenna),'part_2_edge_sup_ lenna','./result/part_2__edge_sup_lenna.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #shapes
    #Edge detection
    startTime_edge_raw_shapes = time.time()
    shapes_mag, shapes_dir=fn.compute_image_gradient(shapes_img)
    endTime_edge_raw_shapes = time.time()
    print("Time consumed for getting edge raw of shapes image is "+str(endTime_edge_raw_shapes-startTime_edge_raw_shapes))
    fn.show_write(fn.normalize_Img(shapes_mag),'part_2_edge_raw_ shapes','./result/part_2_edge_raw_shapes.png')

    #non maximum suprresion
    startTime_NMS_shapes = time.time()
    shapes_mag, shapes_dir=fn.compute_image_gradient(shapes_img)
    endTime_NMS_shapes = time.time()
    NMS_shapes=fn.non_maximum_suppression_dir(shapes_mag,shapes_dir+math.pi)
    print("Time consumed for suppressing gradients of shapes image is "+str(endTime_NMS_shapes-startTime_NMS_shapes))
    fn.show_write(fn.normalize_Img(NMS_shapes),'part_2_edge_sup_ shapes','./result/part_2__edge_sup_shapes.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
if __name__ == "__main__":
    main()
