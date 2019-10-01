import cv2
import numpy as np
import colorshape
import sys
from PIL import Image
print('Please Load image:')
path = '3.png'

class Glycan:
    x,y,w,h = 0,0,0,0
    contour = np.ndarray
    color = ''
    shape = ''
    child = []


def threshold(imageArray):
    newAr = imageArray
    for eachRow in newAr:
        for eachPix in eachRow:
            # if reduce(lambda x, y: x + y, eachPix[:3]) / 3 > balance:
            if sum(eachPix[:3]) / 3 > 200 or sum(eachPix[:3]) / 3 < 60:
                eachPix[0], eachPix[1], eachPix[2] = 255, 255, 255

    for eachRow in newAr:
        for eachPix in eachRow:
            # if reduce(lambda x, y: x + y, eachPix[:3]) / 3 > balance:
            if sum(eachPix[:3]) / 3 > 200:
                eachPix[0], eachPix[1], eachPix[2] = 255, 255, 255
            else:
                eachPix[0], eachPix[1], eachPix[2] = 0, 0, 0
    #cv2.imshow("newAR",newAr)
    return newAr


# Let's load a simple image with 3 black squares
image = cv2.imread(path)
orgimage = Image.open(path)
orgimage2 = cv2.imread(path)
cv2.imshow('Input', image)
cv2.waitKey(0)
print('Processing Image......')
threshold(image)
#print(type(threshold(image)))

image = cv2.GaussianBlur(image, (9,9), 0)
#cv2.imshow("blurry",image)
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(image, 100, 200)
print('Successful.')
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours1, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = []

for contour in contours1:
    #print(type(contour))
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area
    #print(area)
    if w >6 and h >6 and area >15 and extent>0.04:
        contours.append(contour)
        #print(extent)
        cv2.rectangle(orgimage2, (x, y), (x + w, y + h), (0, 0, 255), 1)
        im1 = orgimage.crop((x, y, x+w, y+h))
        color, shape = colorshape.findcolorshape(im1)
        #print(color,shape)
        cv2.putText(orgimage2, color[:4]+''+shape[:4], (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        #im1.show()
        #print(cv2.boundingRect(contour))

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours, other positive value = any of the index in countours
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', orgimage2)
cv2.waitKey(0)
cv2.destroyAllWindows()
