# from package import extractContours, generateBezierCurve, bwlabel, countAreaOfRegion, convertBinaryToPoints
from package import thin_demo, isTconnection, extractControlPoints, convertBinaryToLists, randomDeform, smoothing_base_bezier
import numpy as np
import math
import cv2 as cv
import bezier
import random
import os
import shutil
import matplotlib.pyplot as plt

# delete and create a folder for outputs.
try:
    shutil.rmtree(os.path.join(os.getcwd(), "new_image"))
except FileNotFoundError as error: print("new_image/ folder doesn't exist")
os.mkdir(os.path.join(os.getcwd(), "new_image"))

path = 'archive/Img/'
# path = 'archive/matlab/'
pathname = 'new_image/'
for filename in os.listdir(path):
    name, extension = filename.split(".")
    category, index = name.split("-")
    extension = "." + extension
    ##############
    name = 'img002-006'
    ##############
    # print(name)
    img = cv.imread(path + name + extension)
    row = img.shape[0]
    skel = thin_demo(img)
    plt.figure(100)
    plt.imshow(skel)

        
    ############ parameters to be changed ############
    degreeOfShifting = math.floor(row / 256) # to be modified
    patchResolution = int(row / 256 * 32) # to be modified
    numOfDeform = 3 # to be modified
    ############ parameters to be changed ############  

    for i in range(1, row-1):
        for j in range(1, row-1):
            if skel[i, j]:
                isTconnected = isTconnection(skel[i-1:i+2, j-1:j+2])
                if (isTconnected):
                    skel[i, j] = False
    
    pointList, lengthList = convertBinaryToLists(skel)

    for i in range(len(pointList)):
        # plt.plot(points[1], -points[0], color='black')
        
        control_points = extractControlPoints(pointList[i], degree=3)
        curve = bezier.Curve(control_points, degree=3)
        # print(lengthList[i])
        list_deformedPoints = randomDeform(curve.nodes, numOfDeform, degreeOfShifting * lengthList[i] / 12)
        ind = 0
        for point in list_deformedPoints:
            x_curve, y_curve = smoothing_base_bezier(point[0], point[1], k=0.6, closed=False)
            axs = plt.gca()
            axs.axis("equal")
            axs.set_axis_off()
            plt.figure(ind)
            plt.plot(y_curve, -x_curve, color='black')
            plt.savefig(pathname + name + '-{0:03}'.format(ind) + '.png')
            img = cv.imread(pathname + name + '-{0:03}'.format(ind) + '.png', 0)
            kernel = np.ones((35, 35), 'uint8')
            dilate_img = cv.dilate(255-img, kernel, iterations=1)
            cv.imwrite(pathname + name + '-{0:03}'.format(ind) + '.png', dilate_img)
            ind += 1

    break