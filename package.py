import cv2 as cv
from skimage.morphology import thin, skeletonize
import numpy as np
import bezier
import pandas as pd
import matplotlib.pyplot as plt



def thin_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours = thin(binary)
    
    return contours
# img = cv.imread("/Users/henghong/Downloads/archive/Img/img001-001.png")
# thin_demo(img)


def extractContours(contours):
    contours = contours[0].tolist()
    x = []
    y = []
    newContours = []
    for i in contours:
        newContours.append(i[0])
    for i in newContours:
        x.append(i[0])
        y.append(i[1])
    return np.asfortranarray([x, y])


def extractControlPoints(nodes, degree=3):
    xnew = [nodes[0][0]]
    ynew = [nodes[1][0]]

    for i in range(degree):
        xInd = int((len(nodes[0])/degree)*(i+1))
        yInd = int((len(nodes[1])/degree)*(i+1))
        # print(f"xInd = {xInd}")
        # print(f"yInd = {yInd}")
        xnew.append(nodes[0][xInd-1])
        ynew.append(nodes[1][yInd-1])

    return np.asfortranarray([xnew, ynew])



def generateBezierCurve(fileName, nodes, numSegments=1, filename=None, degree=3, toPlot=False):
    if numSegments < 1:
        print("wrong numSegments argument")
        return
    intervalNodes = int(len(nodes[0]) / (numSegments)) + 1
    j = 0
    if toPlot: plotImg(originalName=fileName, index=j, old_nodes=nodes, ifPlotOldNode=True)
    for i in range(0, len(nodes[0]), intervalNodes):
        x_segment = nodes[0][i:i+intervalNodes].tolist()
        y_segment = nodes[1][i:i+intervalNodes].tolist()
        controlPoints_of_segment = extractControlPoints([x_segment, y_segment], degree)
        curve = bezier.Curve(controlPoints_of_segment, degree=degree)

        # pathname = './new_images/' + filename + '-{0:03}'.format(j) + ".csv"
        # pd.DataFrame([x_new, y_new]).to_csv(pathname)
        j = j + 1

        if toPlot: 
            plotImg(originalName=fileName, index=j, ifPlotOldNode=False, old_nodes=nodes, new_curve=curve, startpoint=i, endpoint=i+intervalNodes)


def plotImg(originalName, index, old_nodes=[0], ifPlotOldNode=False, new_curve=None, startpoint=-1, endpoint=-1): # plot images
    fig = plt.figure(frameon=False)
    axs = plt.gca()
    axs.axis('equal')
    if ifPlotOldNode == True: 
        plt.plot(old_nodes[0], old_nodes[1], '-y') # original image
        pass
    if startpoint != -1 and endpoint != -1: 
        plt.plot(old_nodes[0][0:startpoint], old_nodes[1][0:startpoint], '-r') # unchanged segment in new image
        plt.plot(old_nodes[0][endpoint:len(old_nodes[0])], old_nodes[1][endpoint:len(old_nodes[0])], '-r') # unchanged segment in new image
    if new_curve != None: 
        new_curve.plot(len(new_curve.nodes[0]), ax=axs) # new segment in new image (option 2)
    axs.invert_yaxis()
    pathname = './new_images/'
    plt.savefig(pathname + originalName + '-{0:03}'.format(index) + '.png')
    axs.clear()
    # plt.show()