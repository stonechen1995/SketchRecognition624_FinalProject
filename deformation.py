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

# delete and create a folder for outputs.
try:
    shutil.rmtree(os.path.join(os.getcwd(), "new_image"))
except FileNotFoundError as error: print("new_image/ folder doesn't exist")
os.mkdir(os.path.join(os.getcwd(), "new_image"))

path = 'archive/Img/'
# path = 'archive/matlab/'
pathname = 'new_images/'
for filename in os.listdir(path):
    name, extension = filename.split(".")
    category, index = name.split("-")
    extension = "." + extension
    ##############
    name = 'img011-035'
    ##############
    print(name)
    img = cv.imread(path + name + extension)
    row = img.shape[0]
    skel = thin_demo(img)
    plt.figure(100)
    plt.imshow(skel)

        
    ############ parameters to be changed ############
    degreeOfShifting = math.floor(row/256) * 4 # to be modified
    patchResolution = int(row / 256 * 32) # to be modified
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
    shouldBreak = False
    for i in range(0, int(row/patchResolution)):
        for j in range(0, int(row / patchResolution)):
    # i = 12
    # j = 8

    # 6 9
    # 7 8
    # 8 8
    # 9 9
    # 10 8
    # 11 8
    # 12 8
    # for each patch
            patch = skel[i * patchResolution : (i+1) * patchResolution, j * patchResolution : (j+1) * patchResolution]
            # if (i == 8 and j == 12): 
            #     plt.figure(0)
            #     plt.imshow(patch)
            if patch.any() == True:
                # print('i, j',i, j)

                # break
                # plt.subplot(111)
                # plt.imshow(patch)
                # plt.show()
                pointsInMax, pointRest = convertBinaryToPoints(patch)
                # print(pointRest)

                # if pointsInMax is None:
                #     ind = 1
                #     for _ in range(numOfDeform):
                #         axs = plt.gca()
                #         axs.axis("equal")
                #         axs.set_axis_off()
                #         plt.figure(ind)
                #         plt.plot(pointRest[1] + j * patchResolution, -pointRest[0] - i * patchResolution, color='black')
                #         ind += 1

                # 4 control points from the original segment
                if pointsInMax.shape[1] > 3:
                    controlPoints_of_segment = extractControlPoints(pointsInMax, degree=3) 
                    curve = bezier.Curve(controlPoints_of_segment, degree=3)
                    # 4 Beziered points from the original segment
                    list_deformedPoints = randomDeform(curve.nodes, numOfDeform, degreeOfShifting)
                    ind = 1
                    for point in list_deformedPoints:
                        x = point[0] + i * patchResolution
                        y = point[1] + j * patchResolution
                        
                        x_curve, y_curve = smoothing_base_bezier(x, y, k=0.6, closed=False)


                        axs = plt.gca()
                        axs.axis("equal")
                        axs.set_axis_off()
                        plt.figure(ind)


                        gridx = np.array(list(range(i * patchResolution, (i+1) * patchResolution)))
                        gridy = np.array([j * patchResolution] * patchResolution)
                        plt.plot(gridy, -gridx)

                        gridx = np.array(list(range(i * patchResolution, (i+1) * patchResolution)))
                        gridy = np.array([(j+1) * patchResolution] * patchResolution)
                        plt.plot(gridy, -gridx)

                        gridx = np.array([i * patchResolution] * patchResolution)
                        gridy = np.array(list(range(j * patchResolution, (j+1) * patchResolution)))
                        plt.plot(gridy, -gridx)

                        gridx = np.array([(i+1) * patchResolution] * patchResolution)
                        gridy = np.array(list(range(j * patchResolution, (j+1) * patchResolution)))
                        plt.plot(gridy, -gridx)


                        plt.plot(y_curve, -x_curve, color='black')
                        # plt.plot(pointRest[1] + j * patchResolution, -pointRest[0] - i * patchResolution, color='black')
                        # plt.plot(y, -x, 'ro')
                        plt.savefig(pathname + name + '-{0:03}'.format(ind) + '.png')
                        ind += 1
                        # break
        #     if i == 7 and j == 11:
        #         shouldBreak = True
        #         break
        # if shouldBreak:
        #     break

    break