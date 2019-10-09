import cv2
import numpy as np
import colorshape
import sys
from PIL import Image
import linedetecttest2
print('Please Load image:')
path = '1.png'

class Glycan: # create family for each glycan
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


def findroot(image):

    img = image.copy()
    root = cv2.imread('glycanshape/root.png')
    h,w,unknown1 = root.shape[:]
    #print(w, h,unknown1)
    method = eval('cv2.TM_CCOEFF') #['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # Apply template Matching
    res = cv2.matchTemplate(img, root, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(min_val, max_val, min_loc, max_loc)
    bottom_right = (max_loc[0] + w, max_loc[1] + h)

    #cv2.rectangle(img, max_loc, bottom_right, (255, 0, 255), 1)
    cv2.rectangle(img, max_loc, bottom_right,(255, 255, 255), -1)
    cv2.putText(img, 'Root', (max_loc[0]-10,max_loc[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
                cv2.LINE_AA)
    return img



def findAllConnectionSymAux(image):
    img_rgb = image.copy()
    whitesymbol = image.copy()
    string = '2346aBq'
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    for c in string:
        template = cv2.imread(f'glycanshape/{c}.png', 0)
        h,w = template.shape[:]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
            cv2.rectangle(whitesymbol, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), -1) # fill symb with white space
            cv2.putText(img_rgb, c, (pt[0], pt[1] + 2*h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)
    #cv2.imwrite('res2.png', img_rgb)
    return img_rgb,whitesymbol


def findLines(image,shapes,symbols):
    whiteshapelines = image


    return whiteshapelines

#  load a simple image
image = cv2.imread(path)
orgimage = Image.open(path)
whiteshapelines = image.copy()
finalImage = image.copy()
cv2.imshow('Input', image)
cv2.waitKey(0)
print('Processing Image......')
threshold(image)
cv2.imshow('Applied Threshold',image)
cv2.waitKey(0)

#print(type(threshold(image)))


image = cv2.GaussianBlur(image, (9,9), 0)
#cv2.imshow("blurry",image)
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(image, 100, 200)

print('Successful.')
cv2.waitKey(0)

#fill sym with white
finalImage,whitesymbol=findAllConnectionSymAux(finalImage)
whiteshapelines=findAllConnectionSymAux(whiteshapelines)[1]


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
        cv2.rectangle(finalImage, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(whiteshapelines, (x, y-4), (x + w, y + h+3), (255, 255, 255), -1)#whiteshapelines
        im1 = orgimage.crop((x, y, x+w, y+h))
        color, shape = colorshape.findcolorshape(im1)
        cv2.putText(finalImage, color[:4] + '' + shape[:4], (x - 6, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)


cv2.imshow('Edged', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours, other positive value = any of the index in countours
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

finalImage=findroot(finalImage)

newX,newY = int(finalImage.shape[1] * 1.5), int(finalImage.shape[0] * 1.5)
finalImage=cv2.resize(finalImage, (newX, newY))

#need to process lines
cv2.imwrite("linesonly.png",whiteshapelines)
whiteshapelines = linedetecttest2.findLines(whiteshapelines)
cv2.imshow('only lines', whiteshapelines)
cv2.waitKey(0)


cv2.imshow('Final Image', finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
