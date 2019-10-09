import cv2
import numpy as np
import colorshape
import sys
from PIL import Image
from matplotlib import pyplot as plt

image = cv2.imread('3.png')
cv2.imshow('name',image)
cv2.waitKey(0)
img2 = image.copy()
root = cv2.imread('glycanshape/root.png', 0)
w, h = root.shape[::-1]
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
# th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,0)
# print(type(ret),type(thresh))
img2 = cv2.Canny(img2, 100, 200)
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ['cv2.TM_CCOEFF']
for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, root, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    color = (255, 0, 255)
    cv2.rectangle(image, top_left, bottom_right, color, 1)
    cv2.putText(image, 'Root', min_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1,
                cv2.LINE_AA)

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
cv2.imshow('name',image)
cv2.waitKey(0)
'''
edged = cv2.Canny(image,100,200)
cv2.imshow('Input', edged)
cv2.waitKey(0)'''
