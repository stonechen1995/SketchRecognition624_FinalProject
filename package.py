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


# extract points in the region of max area in the binary image into np.array
def convertBinaryToPoints(binaryMatrix):
    connectedRegionMatrix, numOfConnectedRegions = bwlabel(binaryMatrix)
    List_AreaOfConnectedRegions = countAreaOfRegion(connectedRegionMatrix, numOfConnectedRegions)
    # print(f"List_AreaOfConnectedRegions = {List_AreaOfConnectedRegions}")
    index_maxAreaOfRegions = max(range(len(List_AreaOfConnectedRegions)), key=List_AreaOfConnectedRegions.__getitem__)
    maxArea = List_AreaOfConnectedRegions[index_maxAreaOfRegions]
    x = []
    y = []
    if maxArea > 4: # "4" as a threshold
        # print("maxArea > 4")
        for i in range(0, binaryMatrix.shape[0]):
            for j in range(0, binaryMatrix.shape[0]):
                if connectedRegionMatrix[i, j] == index_maxAreaOfRegions: 
                    # print("connectedRegionMatrix[i, j] == index_maxAreaOfRegions")
                    x.append(i)
                    y.append(j)
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


def isTconnection(matrix):
    if sum(matrix[0][:])==3 or sum(matrix[2][:])==3 or sum(matrix[:][0])==3 or sum(matrix[:][2])==3:
        return True
    return False


def bwlabel(inputMatrix):
    label = 0
    res = 1 * inputMatrix
    flag = inputMatrix
    x = len(res)
    y = len(res[0])
    for i in range(x):
        for j in range(y):
            if not flag[i, j]: continue
            label += 1
            res[i, j] = label
            DFS(res, i, j, flag)
    return res, label

def DFS(res, i, j, flag):
    flag[i, j] = 0
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
    areaOfEachRegion = list(0 for _ in range(numOfConnectedRegions+1))
    for i in range(0, row):
        for j in range(0, row):
            if inputMatrix[i, j]:
                # print(f"numOfConnectedRegions = {numOfConnectedRegions}; inputMatrix[i, j] = {inputMatrix[i, j]}")
                areaOfEachRegion[inputMatrix[i, j]] = areaOfEachRegion[inputMatrix[i, j]] + 1
    return areaOfEachRegion


# input  [[x0,x1,x2,x3],[y0,y1,y2,y3]] 
# output [[[x0,x1,x2,x3],[y0,y1,y2,y3]],   [[x0,x1,x2,x3],[y0,y1,y2,y3]],   [[x0,x1,x2,x3][y0,y1,y2,y3]],   [[x0,x1,x2,x3][y0,y1,y2,y3]], ....]
def randomDeform(node, num_deform, degree):
    res = []
    for _ in range(num_deform):
        tmp = node.copy()
        shift = np.random.rand(2, 2) * (degree * 2) - degree
        tmp[:, 1:3] += shift.astype(int)
        # print(tmp)
        res.append(tmp)
    return res

def bezier_curve(p0, p1, p2, p3, inserted):
    """
    三阶贝塞尔曲线
    
    p0, p1, p2, p3  - 点坐标，tuple、list或numpy.ndarray类型
    inserted        - p0和p3之间插值的数量
    """
    
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    
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
    """
    基于三阶贝塞尔曲线的数据平滑算法
    
    date_x      - x维度数据集，list或numpy.ndarray类型
    date_y      - y维度数据集，list或numpy.ndarray类型
    k           - 调整平滑曲线形状的因子，取值一般在0.2~0.6之间。默认值为0.5
    inserted    - 两个原始数据点之间插值的数量。默认值为10
    closed      - 曲线是否封闭，如是，则首尾相连。默认曲线不封闭
    """
    
    assert isinstance(date_x, (list, np.ndarray)), u'x数据集不是期望的列表或numpy数组类型'
    assert isinstance(date_y, (list, np.ndarray)), u'y数据集不是期望的列表或numpy数组类型'
    
    if isinstance(date_x, list) and isinstance(date_y, list):
        assert len(date_x)==len(date_y), u'x数据集和y数据集长度不匹配'
        date_x = np.array(date_x)
        date_y = np.array(date_y)
    elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
        assert date_x.shape==date_y.shape, u'x数据集和y数据集长度不匹配'
    else:
        raise Exception(u'x数据集或y数据集类型错误')
    
    # 第1步：生成原始数据折线中点集
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
    
    # 第2步：找出中点连线及其分割点
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
    
    # 第3步：平移中点连线，调整端点，生成控制点
    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end'] # 当前顶点的坐标
        dx = vx - split_points[i]['split'][0] # 平移线段x偏移量
        dy = vy - split_points[i]['split'][1] # 平移线段y偏移量
        
        sx, sy = split_points[i]['start'][0]+dx, split_points[i]['start'][1]+dy # 平移后线段起点坐标
        ex, ey = split_points[i]['end'][0]+dx, split_points[i]['end'][1]+dy # 平移后线段终点坐标
        
        cp0 = sx+(vx-sx)*k, sy+(vy-sy)*k # 控制点坐标
        cp1 = ex+(vx-ex)*k, ey+(vy-ey)*k # 控制点坐标
        
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
    
    # 第4步：应用贝塞尔曲线方程插值
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