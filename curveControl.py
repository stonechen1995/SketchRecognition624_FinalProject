from package import thin_demo
from package import extractContours
from package import generateBezierCurve
import cv2 as cv
import random
import os
import shutil


# delete and create a folder for outputs.
try:
    shutil.rmtree(os.path.join(os.getcwd(), "new_images"))
except FileNotFoundError as error: print("new_images/ folder doesn't exist")
os.mkdir(os.path.join(os.getcwd(), "new_images"))

selected = set()
for i in range(5):
   selected.add('{0:03}'.format(random.randint(1,55)))

path = 'archive/Img/'
for filename in os.listdir(path):
    name, extension = filename.split(".")
    category, index = name.split("-")
    if index not in selected: continue
    extension = "." + extension
    ##############
    name = 'img058-047'
    ##############
    print(name)
    img = cv.imread(path + name + extension)
    contours = thin_demo(img)
    # print(contours)
    nodes = extractContours(contours)
    generateBezierCurve(name, nodes, numSegments=8, filename=name, degree=3, toPlot=True)
    break