# procedure:
# break t-connection。
# 分patch。
# initialize 9组list， list0 — list8
# 每个patch，
# 1. 找最大的连起来的区域的所有点，其他的归0。
# 2. 把找到的这块区域所有点存到一个temp list, 再找到这个temp list其中4个control points。将这4个点的绝对坐标存进list0.
# 3. 把这个temp list 喂到bezier里，得到4个点，再进行8组shift得到另外8组4个点。将这4个点的绝对坐标分别存进list1 — list8
# 4. 用polyline function分别连list0 — list8中所有的点, https://www.tutorialspoint.com/get-the-least-squares-fit-of-a-polynomial-to-data-in-python
# 5. 
# 存下来。


from package import thin_demo, extractContours, generateBezierCurve, isTconnection, bwlabel, countAreaOfRegion, extractControlPoints, convertBinaryToPoints, randomDeform, smoothing_base_bezier
import numpy as np
import math
import cv2 as cv
import bezier
import random
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

path = 'archive/Img/'
pathname = 'new_images/'
for filename in os.listdir(path):
    name, extension = filename.split(".")
    category, index = name.split("-")
    extension = "." + extension
    ##############
    name = 'img009-033'
    ##############
    print(name)
    img = cv.imread(path + name + extension)
    row = img.shape[0]
    skel = thin_demo(img)
    plt.figure(100)
    plt.imshow(skel)

        
    ############ parameters to be changed ############
    degreeOfShifting = math.floor(row/256) * 1 # to be modified
    patchResolution = int(row / 256 * 4) # to be modified
    numOfDeform = 3 # to be modified
    ############ parameters to be changed ############

    for i in range(1, row-1):
        for j in range(1, row-1):
            if skel[i, j]:
                isTconnected = isTconnection(skel[i-1:i+2, j-1:j+2])
                if (isTconnected):
                    skel[i, j] = False
    
    mapsOfPotins = {} # to store points
    for i in range(1, numOfDeform+1): mapsOfPotins[i] = [[],[]]

    for i in range(0, int(row/patchResolution)):
        for j in range(0, int(row / patchResolution)):
            # for each patch
            patch = skel[i * patchResolution : (i+1) * patchResolution, j * patchResolution : (j+1) * patchResolution]
            if patch.any() == True:
                # plt.subplot(111)
                # plt.imshow(patch)
                # plt.show()
                pointsInPatch = convertBinaryToPoints(patch)

                # 4 control points from the original segment
                if pointsInPatch.shape[1] > 4:
                    controlPoints_of_segment = extractControlPoints(pointsInPatch, degree=3) 
                curve = bezier.Curve(controlPoints_of_segment, degree=3)
                # 4 Beziered points from the original segment
                list_deformedPoints = randomDeform(curve.nodes, numOfDeform, degreeOfShifting)
                ind = 1
                for point in list_deformedPoints:
                    x = point[0] + i * patchResolution
                    y = point[1] + j * patchResolution

                    axs = plt.gca()
                    axs.axis("equal")
                    plt.figure(ind)
                    x_curve, y_curve = smoothing_base_bezier(x, y, k=0.6, closed=False)



                    # gridx = np.array(list(range(i * patchResolution, (i+1) * patchResolution)))
                    # gridy = np.array([j * patchResolution] * patchResolution)
                    # plt.plot(gridy, -gridx)

                    # gridx = np.array(list(range(i * patchResolution, (i+1) * patchResolution)))
                    # gridy = np.array([(j+1) * patchResolution] * patchResolution)
                    # plt.plot(gridy, -gridx)

                    # gridx = np.array([i * patchResolution] * patchResolution)
                    # gridy = np.array(list(range(j * patchResolution, (j+1) * patchResolution)))
                    # plt.plot(gridy, -gridx)

                    # gridx = np.array([(i+1) * patchResolution] * patchResolution)
                    # gridy = np.array(list(range(j * patchResolution, (j+1) * patchResolution)))
                    # plt.plot(gridy, -gridx)





                    plt.plot(y_curve, -x_curve, label='$k=0.3$')
                    # plt.plot(y, -x, 'ro')
                    plt.savefig(pathname + name + '-{0:03}'.format(ind) + '.png')
                    ind += 1

    break