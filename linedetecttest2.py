import cv2
import numpy as np
'''
img = cv2.imread('image.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('1',edges)
cv2.waitKey(0)
minLineLength = 0
maxLineGap = 0
lines = cv2.
lines =cv2.HoughLines(edges, 1, np.pi / 180, 200)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)
'''
def findLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('1', thresh)
    cv2.waitKey(0)
    rho = 1
    theta = np.pi/180
    threshold = 13#23 13 work best
    min_line_length = 10
    max_line_gap = 1
    lines = cv2.HoughLinesP(thresh, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    print(lines)
    print(len(lines))
    count=0
    for lines[0][0] in lines:
        x1, y1, x2, y2 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3],
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, 'p1', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(img, 'p2', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),
                    1, cv2.LINE_AA)
        count=count+1
        print(count)
    return img
'''img = cv2.imread('linesonly.png')
img = findLines(img)
cv2.imshow('24', img)
cv2.waitKey(0)'''