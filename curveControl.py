from package import thin_demo
from package import extractContours
from package import generateBezierCurve
import cv2 as cv
import os

# create a folder for outputs.
try:
    os.mkdir(os.path.join(os.getcwd(), "new_images"))
except FileExistsError as error: print("new_images/ folder exist")


path = 'archive/Img/'
for filename in os.listdir(path):
    name, extension = filename.split(".")
    extension = "." + extension
    img = cv.imread(path + name + extension)
    contours = thin_demo(img)
    nodes = extractContours(contours)
    generateBezierCurve(nodes, numSegments=8, filename=name, toPlot=False)
    