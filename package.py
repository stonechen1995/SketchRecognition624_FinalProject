import cv2 as cv
import numpy as np
import bezier
import pandas as pd
import matplotlib.pyplot as plt



def thin_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    thinned = cv.ximgproc.thinning(binary)
    contours, hireachy = cv.findContours(thinned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(image, contours, -1, (0, 0, 255), 1, 8)
    # cv.imshow("thin", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return contours
# img = cv.imread("/Users/henghong/Downloads/archive/Img/img001-001.png")
# thin_demo(img)


def extractContours(contours):
    x = []
    y = []
    for i in contours:
        for j in i:
            for k in j:
                x.append(k[0])
                y.append(k[1])

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
    if toPlot: plotImg(originalName=fileName, index=j, old_nodes=nodes)
    for i in range(0, len(nodes[0]), intervalNodes):
        print(f"\n\ni={i}")
        x_new = nodes[0].tolist()
        y_new = nodes[1].tolist()
        print(f"x_new = {x_new}")
        x_segment = x_new[i:i+intervalNodes]
        y_segment = y_new[i:i+intervalNodes]
        print(f"len(x_segment) = {len(x_segment)}")
        controlPoints_of_segment = extractControlPoints([x_segment, y_segment], degree)
        print(f"len(controlPoint) = {len(controlPoints_of_segment[0])}")
        curve = bezier.Curve(controlPoints_of_segment, degree=degree)
        print(f"curve = {curve.nodes[0]}")
        x_new[i:i+intervalNodes] = curve.nodes[0].tolist()
        y_new[i:i+intervalNodes] = curve.nodes[1].tolist()
        print(f"len(x_new) = {len(x_new)}")
        # pathname = './new_images/' + filename + '-{0:03}'.format(j) + ".csv"
        # pd.DataFrame([x_new, y_new]).to_csv(pathname)
        j = j + 1

        if toPlot: 
            # plotImg(old_nodes=nodes, new_curve=curve, new_nodes=[x_new, y_new], startpoint=i, endpoint=i+intervalNodes)
            # plotImg(originalName=fileName, index=j, old_nodes=nodes, new_curve=curve)
            # plotImg(originalName=fileName, index=j, new_curve=curve, new_nodes=[x_new, y_new], startpoint=i, endpoint=i+intervalNodes)
            # plotImg(originalName=fileName, index=j, new_nodes=[x_new, y_new], startpoint=i, endpoint=i+intervalNodes)
            # plotImg(originalName=fileName, index=j+10, new_nodes=[x_new, y_new])
            # plotImg(originalName=fileName, index=j+10, old_nodes=nodes, new_curve=curve)
            plotImg(originalName=fileName, index=j, new_nodes=[x_new, y_new], new_curve=curve, startpoint=i, endpoint=i+intervalNodes)


def plotImg(originalName, index, old_nodes=[0], new_curve=None, new_nodes=None, startpoint=-1, endpoint=-1): # plot images
    axs = plt.gca()
    axs.axis('equal')
    if len(old_nodes) > 1: 
        plt.plot(old_nodes[0], old_nodes[1], '-y') # original image
    if new_nodes != None: 
        if startpoint == -1 or endpoint == -1: 
            plt.plot(new_nodes[0], new_nodes[1], '-b')
        else: 
            print(f"startpoint={startpoint}; endpoint={endpoint}")
            plt.plot(new_nodes[0][0:startpoint], new_nodes[1][0:startpoint], '-r') # unchanged segment in new image
            plt.plot(new_nodes[0][endpoint:len(new_nodes[0])], new_nodes[1][endpoint:len(new_nodes[0])], '-r') # unchanged segment in new image
            plt.plot(new_nodes[0][startpoint:endpoint], new_nodes[1][startpoint:endpoint], '--y') # new segment in new image (option 1)
    if new_curve != None: 
        # new_curve.plot(len(new_curve.nodes[0]), ax=axs) # new segment in new image (option 2, different from option 1 and option 3)
        plt.plot(new_curve.nodes[0], new_curve.nodes[1], '--b') #option 3
    axs.invert_yaxis()
    pathname = './new_images/'
    plt.savefig(pathname + originalName + '-{0:03}'.format(index) + '.png')
    axs.clear()
    # plt.show()