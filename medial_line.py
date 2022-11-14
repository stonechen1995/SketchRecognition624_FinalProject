import cv2 as cv
import numpy as np


def thin_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    thinned = cv.ximgproc.thinning(binary)
    contours, hireachy = cv.findContours(thinned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 0, 255), 1, 8)
    cv.imshow("thin", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
img = cv.imread("/Users/ruichenni/Downloads/archive/Img/img009-022.png")
thin_demo(img)
