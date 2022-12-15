import cv2 as cv
from skimage.morphology import thin, skeletonize
import numpy as np
import bezier
import pandas as pd
import matplotlib.pyplot as plt
import sys
print(sys.setrecursionlimit(2000))
print(sys.getrecursionlimit())

label_list = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]

def thin_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours = thin(binary)
    
    return contours # binary matrix
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

def convertBinaryToLists(binaryMatrix):
    connectedRegionMatrix, numOfConnectedRegions = bwlabel(binaryMatrix)
    plt.figure(10)
    plt.imshow(connectedRegionMatrix)
    print('numOfConnectedRegions', numOfConnectedRegions)

    List_AreaOfConnectedRegions = countAreaOfRegion(connectedRegionMatrix, numOfConnectedRegions)
    print('List_AreaOfConnectedRegions', List_AreaOfConnectedRegions)

    connectedRegionMatrixWithZeros = np.zeros((binaryMatrix.shape[0]+2, binaryMatrix.shape[1]+2))
    connectedRegionMatrixWithZeros[1:binaryMatrix.shape[0]+1, 1:binaryMatrix.shape[1]+1] = connectedRegionMatrix
    # plt.figure(100)
    # plt.imshow(connectedRegionMatrixWithZeros)

    lists = []
    for k in range(0, numOfConnectedRegions):
        if List_AreaOfConnectedRegions[k] <= 3: continue
        p0 = None
        shouldBreak = False
        label = label_list[k]
        # print(label)
        tmp = []

        for i in range(0, binaryMatrix.shape[0]):
            for j in range(0, binaryMatrix.shape[0]):
                if connectedRegionMatrix[i, j] != label: continue
                tmp = [i, j]
                if isEndPoint(connectedRegionMatrixWithZeros[i:i+3, j:j+3], label):
                    if p0 == None:
                        p0 = [i, j]
                        print('p0', p0)
                        shouldBreak = True
                        break
            if shouldBreak: break
        connectedRegionMatrixWithZeros[tmp[0]+1, tmp[1]+1] = 0
        if p0 is None:
            p0 = tmp
        points = findPoints(p0, connectedRegionMatrixWithZeros, label)
        l = len(points[0])
        print(l)
        threshold = 75
        num = 0
        while l - threshold * (num+1) > 0:
            lists.append([points[0][num*threshold: (num+1)*threshold], points[1][num*threshold: (num+1)*threshold]])
            num += 1
        lists.append([points[0][num*threshold: l], points[1][num*threshold: l]])
    return lists


# extract points in the region of max area in the binary image into np.array
def convertBinaryToPoints(binaryMatrix):
    connectedRegionMatrix, numOfConnectedRegions = bwlabel(binaryMatrix)
    # print(numOfConnectedRegions)

    List_AreaOfConnectedRegions = countAreaOfRegion(connectedRegionMatrix, numOfConnectedRegions)
    Dict_Regions = {0: 0}
    for i in range(1, len(List_AreaOfConnectedRegions)+1):
        Dict_Regions[i] = List_AreaOfConnectedRegions[i]

    index_maxAreaOfRegions = max(range(len(List_AreaOfConnectedRegions)), key=List_AreaOfConnectedRegions.__getitem__)
    maxArea = List_AreaOfConnectedRegions[index_maxAreaOfRegions]

    print('List_AreaOfConnectedRegions: ', List_AreaOfConnectedRegions)
    print('index_maxAreaOfRegions: ', index_maxAreaOfRegions)
    print('maxArea: ', maxArea)

    p0 = None
    connectedRegionMatrixWithZeros = np.zeros((binaryMatrix.shape[0]+2, binaryMatrix.shape[0]+2))
    x0 = []
    y0 = []
    for i in range(binaryMatrix.shape[0]):
        for j in range(binaryMatrix.shape[1]):
            if connectedRegionMatrix[i, j] == index_maxAreaOfRegions: 
                connectedRegionMatrix[i, j] = 1
            elif connectedRegionMatrix[i, j] != 0:
                connectedRegionMatrix[i, j] = 0
                x0.append(i)
                y0.append(j)
            

    # plt.figure(1000)
    # plt.imshow(connectedRegionMatrix)
    connectedRegionMatrixWithZeros[1:binaryMatrix.shape[0]+1, 1:binaryMatrix.shape[0]+1] = connectedRegionMatrix
    x = []
    y = []
    if maxArea > 4: # "4" as a threshold
        shouldBreak = False
        for i in range(0, binaryMatrix.shape[0]):
            for j in range(0, binaryMatrix.shape[0]):
                if connectedRegionMatrix[i, j] == 1: 
                    # print(i, j)
                    if isEndPoint(connectedRegionMatrixWithZeros[i:i+3, j:j+3]):
                        if p0 == None:
                            p0 = [i, j]
                            connectedRegionMatrixWithZeros[i+1, j+1] = 0
                            # plt.figure(10000)
                            # plt.imshow(connectedRegionMatrixWithZeros)
                            points = findPoints(p0, connectedRegionMatrixWithZeros)
                            if points is None:
                                print("two pixels")
                                for i in range(binaryMatrix.shape[0]):
                                    for j in range(binaryMatrix.shape[1]):
                                        if connectedRegionMatrix[i, j] == 1: 
                                            x0.append(i)
                                            y0.append(j)
                            shouldBreak = True
                            # print('p0', p0)
                            # print(points)
                            break
            if shouldBreak: break
                    #     else: 
                    #         p3 = [i, j]
                    # else:
                    #     x.append(i)
                    #     y.append(j)
                    
        # print(f"{points[0]}, {points[len(points)-1]}")
        if points is None:
            return None, np.asfortranarray([x0, y0])
        for point in points:
            x.append(point[0])
            y.append(point[1])
    return np.asfortranarray([x, y]), np.asfortranarray([x0, y0])


def findPoints(point, graph, label):
    # for each axis neighbor of point, 
        # check if 1, 
            # yes, then store into p and then index point to this point with 1; break the for loop
    # if axis neighbor is not found, check diaganol points
        # check if 1, 
            # yes, then store into p and then index point to this point with 1; break the for loop

    # print("start find points")
    x_list = np.array([point[0]])
    y_list = np.array([point[1]])
    x = point[0] + 1
    y = point[1] + 1
    newpoint = [x, y]
    # print(label)
    # print(newpoint)
    # plt.figure(10000)
    # plt.imshow(graph)
    # print(graph[newpoint[0]-1, newpoint[1]-1])
    # print(graph[newpoint[0]+1, newpoint[1]-1])
    # print(graph[newpoint[0]-1, newpoint[1]+1])
    # print(graph[newpoint[0]+1, newpoint[1]+1])
    while x in range(graph.shape[0]) and y in range(graph.shape[1]):
        # if graph[x-1,y-1] and graph[x-1, y] and graph[x, y-1]: return None
        # if graph[x-1,y+1] and graph[x-1, y] and graph[x, y+1]: return None
        # if graph[x+1,y-1] and graph[x+1, y] and graph[x, y-1]: return None
        # if graph[x+1,y+1] and graph[x+1, y] and graph[x, y+1]: return None
        
        if graph[x, y-1] == label: 
            newpoint = [x, y-1]
        elif graph[x, y+1] == label:
            newpoint = [x, y+1]
        elif graph[x-1, y] == label:
            newpoint = [x-1, y]
        elif graph[x+1, y] == label:
            newpoint = [x+1, y]
        elif graph[x-1, y-1] == label: 
            newpoint = [x-1, y-1]
        elif graph[x+1, y+1] == label:
            newpoint = [x+1, y+1]
        elif graph[x-1, y+1] == label:
            newpoint = [x-1, y+1]
        elif graph[x+1, y-1] == label:
            newpoint = [x+1, y-1]
        else:
            break
        graph[x, y] = 0
        x = newpoint[0]
        y = newpoint[1]
        x_list = np.append(x_list, x - 1)
        y_list = np.append(y_list, y - 1)
    return np.asfortranarray([x_list, y_list])
        
    # while True:
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             x = newpoint[0]
    #             y = newpoint[1]
    #             flag = False
    #             if abs(i) + abs(j) == 1:
    #                 if connectedRegionMatrix[x + 1 + i, y + 1 + j]: 
    #                     newpoint = [x + i, y + j]
    #                     connectedRegionMatrix[x, y] = 0
    #                     p = np.append(p, newpoint)
    #                     continue
    #             if abs(i) + abs(j) == 2:
    #                 if connectedRegionMatrix[x + 1 + i, y + 1 + j]: 
    #                     newpoint = [x + i, y + j]
    #                     connectedRegionMatrix[x, y] = 0
    #                     p = np.append(p, newpoint)
                        # 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 0 0
                        # 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0
                        # 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        # 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        

    
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


def isTconnection(matrix):
    if sum(matrix[0][:])==3 or sum(matrix[2][:])==3 or sum(matrix[:][0])==3 or sum(matrix[:][2])==3:
        return True
    # if matrix[0][0] and matrix[0][2] and sum(matrix[2][:]) > 0
    if sum(matrix[0][:])==2 and matrix[0][1]==0 and sum(matrix[2][:])>0:
        return True
    if sum(matrix[2][:])==2 and matrix[2][1]==0 and sum(matrix[0][:])>0:
        return True
    if sum(matrix[:, 0])==2 and matrix[1][0]==0 and sum(matrix[:, 2])>0:
        return True
    if sum(matrix[:, 2])==2 and matrix[1][2]==0 and sum(matrix[:, 0])>0:
        return True
    if matrix[0,0] and matrix[1,2] and matrix[2,1]:
        return True
    if matrix[0,2] and matrix[1,0] and matrix[2,1]:
        return True
    if matrix[2,0] and matrix[0,1] and matrix[1,2]:
        return True
    if matrix[2,2] and matrix[1,0] and matrix[0,1]:
        return True
    return False

    # 1 0 1
    # 0 1 0
    # 0 1 0 
    # 0 1 0
    # 0 1 0 
    # 0 1 0
    # 0 1 0 
    # 0 1 0
    # 0 1 0 



    # 1 1 1
    # 0 1 0
    # 0 0 0


def bwlabel(inputMatrix):
    num = 0
    res = 1 * inputMatrix # convert boolean to 1 or 0
    flag = inputMatrix.copy()
    x = len(res)
    y = len(res[0])
    for i in range(x):
        for j in range(y):
            if not flag[i, j]: continue
            res[i, j] = label_list[num]
            DFS(res, i, j, flag)
            num += 1
    return res, num


def DFS(res, i, j, flag):
    flag[i, j] = 0
    # print(sum(sum(flag)))
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            # if abs(x) + abs(y) > 1 or x == y: continue
            if x == y == 0: continue
            if i + x < 0 or i + x >= len(res) or j + y < 0 or j + y >= len(res[0]): continue
            if flag[i + x, j + y] and res[i + x, j + y] <= res[i, j]:
                res[i + x, j + y] = res[i, j]
                DFS(res, i + x, j + y, flag)


def countAreaOfRegion(inputMatrix, numOfConnectedRegions):
    row = inputMatrix.shape[0]
    areaOfEachRegion = list(0 for _ in range(numOfConnectedRegions))
    for i in range(0, row):
        for j in range(0, row):
            if inputMatrix[i, j]:
                # print(f"numOfConnectedRegions = {numOfConnectedRegions}; inputMatrix[i, j] = {inputMatrix[i, j]}")
                areaOfEachRegion[label_list.index(inputMatrix[i, j])] += 1
    return areaOfEachRegion


# input  [[x0,x1,x2,x3],[y0,y1,y2,y3]] 
# output [[[x0,x1,x2,x3],[y0,y1,y2,y3]],   [[x0,x1,x2,x3],[y0,y1,y2,y3]],   [[x0,x1,x2,x3][y0,y1,y2,y3]],   [[x0,x1,x2,x3][y0,y1,y2,y3]], ....]
def randomDeform(node, num_deform, degree):
    res = [node]
    # print(res)
    for _ in range(num_deform):
        tmp = node.copy()
        shift = np.random.rand(2, 2) * (degree * 2) - degree
        tmp[:, 1:3] += shift.astype(int)
        # print(tmp)
        res.append(tmp)
    # print(res)
    return res

def bezier_curve(p0, p1, p2, p3, inserted):
    assert isinstance(p0, (tuple, list, np.ndarray)), u'the coordinates of point is not expected type of tuple, list or numpy.array'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'the coordinates of point is not expected type of tuple, list or numpy.array'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'the coordinates of point is not expected type of tuple, list or numpy.array'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'the coordinates of point is not expected type of tuple, list or numpy.array'
    
    if isinstance(p0, (tuple, list)):
        p0 = np.array(p0)
    if isinstance(p1, (tuple, list)):
        p1 = np.array(p1)
    if isinstance(p2, (tuple, list)):
        p2 = np.array(p2)
    if isinstance(p3, (tuple, list)):
        p3 = np.array(p3)
    
    points = list()
    for t in np.linspace(0, 1, inserted+2):
        points.append(p0*np.power((1-t),3) + 3*p1*t*np.power((1-t),2) + 3*p2*(1-t)*np.power(t,2) + p3*np.power(t,3))
    
    return np.vstack(points)


def smoothing_base_bezier(date_x, date_y, k=0.5, inserted=10, closed=False):
    assert isinstance(date_x, (list, np.ndarray)), u'The list of x is not expected type of list or numpy.array'
    assert isinstance(date_y, (list, np.ndarray)), u'The list of y is not expected type of list or numpy.array'
    
    if isinstance(date_x, list) and isinstance(date_y, list):
        assert len(date_x)==len(date_y), u'The lenths of x and y are not matched'
        date_x = np.array(date_x)
        date_y = np.array(date_y)
    elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
        assert date_x.shape==date_y.shape, u'The lenths of x and y are not matched'
    else:
        raise Exception(u'Wrong type of x or y')
    
    # step 1: generate points data on the stroke
    mid_points = list()
    for i in range(1, date_x.shape[0]):
        mid_points.append({
            'start':    (date_x[i-1], date_y[i-1]),
            'end':      (date_x[i], date_y[i]),
            'mid':      ((date_x[i]+date_x[i-1])/2.0, (date_y[i]+date_y[i-1])/2.0)
        })
    
    if closed:
        mid_points.append({
            'start':    (date_x[-1], date_y[-1]),
            'end':      (date_x[0], date_y[0]),
            'mid':      ((date_x[0]+date_x[-1])/2.0, (date_y[0]+date_y[-1])/2.0)
        })
    
    # step 2: find the middle point and its split points.
    split_points = list()
    for i in range(len(mid_points)):
        if i < (len(mid_points)-1):
            j = i+1
        elif closed:
            j = 0
        else:
            continue
        
        x00, y00 = mid_points[i]['start']
        x01, y01 = mid_points[i]['end']
        x10, y10 = mid_points[j]['start']
        x11, y11 = mid_points[j]['end']
        d0 = np.sqrt(np.power((x00-x01), 2) + np.power((y00-y01), 2))
        d1 = np.sqrt(np.power((x10-x11), 2) + np.power((y10-y11), 2))
        k_split = 1.0*d0/(d0+d1)
        
        mx0, my0 = mid_points[i]['mid']
        mx1, my1 = mid_points[j]['mid']
        
        split_points.append({
            'start':    (mx0, my0),
            'end':      (mx1, my1),
            'split':    (mx0+(mx1-mx0)*k_split, my0+(my1-my0)*k_split)
        })
    
    # step 3: move the middle point; adjust the endpoints; generate control points
    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end'] # current endpoint
        dx = vx - split_points[i]['split'][0] # move x of the split point
        dy = vy - split_points[i]['split'][1] # move y of the split point
        
        sx, sy = split_points[i]['start'][0]+dx, split_points[i]['start'][1]+dy # move the starting point of the next segment
        ex, ey = split_points[i]['end'][0]+dx, split_points[i]['end'][1]+dy # move the end point of the next segment
        
        cp0 = sx+(vx-sx)*k, sy+(vy-sy)*k # control point
        cp1 = ex+(vx-ex)*k, ey+(vy-ey)*k # control point
        
        if crt_points:
            crt_points[-1].insert(2, cp0)
        else:
            crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])
        
        if closed:
            if i < (len(mid_points)-1):
                crt_points.append([mid_points[i+1]['start'], cp1, mid_points[i+1]['end']])
            else:
                crt_points[0].insert(1, cp1)
        else:
            if i < (len(mid_points)-2):
                crt_points.append([mid_points[i+1]['start'], cp1, mid_points[i+1]['end']])
            else:
                crt_points.append([mid_points[i+1]['start'], cp1, mid_points[i+1]['end'], mid_points[i+1]['end']])
                crt_points[0].insert(1, mid_points[0]['start'])
    
    # step 4: apply Bezier Function for interpolation
    out = list()
    for item in crt_points:
        group = bezier_curve(item[0], item[1], item[2], item[3], inserted)
        out.append(group[:-1])
    
    out.append(group[-1:])
    out = np.vstack(out)
    
    return out.T[0], out.T[1]

    # usage:

    # if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    
    # x = np.array([2,4,4,3,2])
    # y = np.array([2,2,4,3,4])
	
	# plt.plot(x, y, 'ro')
    # x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=True)
    # plt.plot(x_curve, y_curve, label='$k=0.3$')
    # x_curve, y_curve = smoothing_base_bezier(x, y, k=0.4, closed=True)
    # plt.plot(x_curve, y_curve, label='$k=0.4$')
    # x_curve, y_curve = smoothing_base_bezier(x, y, k=0.5, closed=True)
    # plt.plot(x_curve, y_curve, label='$k=0.5$')
    # x_curve, y_curve = smoothing_base_bezier(x, y, k=0.6, closed=True)
    # plt.plot(x_curve, y_curve, label='$k=0.6$')
    # plt.legend(loc='best')
    
    # plt.show()


def isEndPoint(matrix, label):
    # print("start end point")
    # print(f"{matrix}")
    # 0 1 0 
    # 1 1 0
    # 0 0 0

    # 0 0 0 
    # 1 1 1
    # 0 0 0
    
    # 0 0 1 
    # 0 1 0
    # 0 0 1

    # 0 0 1 
    # 0 1 0
    # 0 0 1
    # print('sum', matrix.sum())

    if matrix.sum() == 2 * label:
        return True

    if matrix.sum() == 3:
        if sum(matrix[0][:])==2 * label and matrix[0][1] == label:
            return True
        if sum(matrix[2][:])==2 * label and matrix[2][1] == label:
            return True
        if sum(matrix[:][0])==2 * label and matrix[1][0] == label:
            return True
        if sum(matrix[:][2])==2 * label and matrix[1][2] == label:
            return True

    return False
