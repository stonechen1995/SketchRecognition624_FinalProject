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


def generateBezierCurve(nodes, numSegments=1, filename=None, toPlot=False):
    if numSegments < 1:
        print("wrong numSegments argument")
        return
        
    intervalNodes = int(len(nodes[0]) / (numSegments)) + 1
    j = 1
    for i in range(0, len(nodes[0]), intervalNodes):
        # print(f"i = {i}")
        x_new = nodes[0]
        y_new = nodes[1]
        x_segment = nodes[0][i:i+intervalNodes]
        y_segment = nodes[1][i:i+intervalNodes]
        nodes_segment = np.asfortranarray([x_segment, y_segment])
        curve = bezier.Curve(nodes_segment, degree=len(nodes_segment[0])-1)
        x_new[i:i+intervalNodes] = curve.nodes[0]
        y_new[i:i+intervalNodes] = curve.nodes[1]
        pathname = './new_images/' + filename + '-{0:03}'.format(j) + ".csv"
        pd.DataFrame([x_new, y_new]).to_csv(pathname)
        j = j + 1

        if toPlot: 
            # plotImg(old_nodes=nodes, new_curve=curve, new_nodes=[x_new, y_new], startpoint=i, endpoint=i+intervalNodes)
            plotImg(old_nodes=nodes, new_curve=curve)


def plotImg(old_nodes=None,  new_curve=None, new_nodes=None, startpoint=0, endpoint=0): # plot images
    axs = plt.gca()
    axs.axis('equal')
    if old_nodes.all() != None: 
        plt.plot(old_nodes[0], old_nodes[1], '-r') # original image
    if new_nodes != None: 
        plt.plot(new_nodes[0][0:startpoint], new_nodes[1][0:startpoint], '-b') # unchanged segment in new image
        plt.plot(new_nodes[0][startpoint:endpoint], new_nodes[1][startpoint:endpoint], '.r') # new segment in new image (option 1)
        plt.plot(new_nodes[0][endpoint:len(new_nodes[0])], new_nodes[1][endpoint:len(new_nodes[0])], '-b') # unchanged segment in new image
    if new_curve != None: 
        new_curve.plot(len(new_curve.nodes[0]), ax=axs) # new segment in new image (option 2)
    plt.show()