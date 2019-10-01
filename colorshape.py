from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import reduce
import sys
import cv2


################ input file here #####################
# path = 'images/numbers/y0.4.png'
# path = 'glycan/red square.png'
# path = 'glycan/complex.png'
# path = 'glycan/yellow.png'
# path = 'glycan/perfectCG.png'
# path = 'glycan/greencircle.png'
# path = 'glycan/triangle.png'
# path = 'glycan/perfectsquare22x22.png'
# path = 'glycan/diamondpurple.png'
# path = 'glycan/bigsquare.png'
# path = 'glycan/redtriangleup.png'

################# input end #####################


def threshold(imageArray):
    newAr = imageArray
    for eachRow in newAr:
        for eachPix in eachRow:
            # if reduce(lambda x, y: x + y, eachPix[:3]) / 3 > balance:
            if sum(eachPix[:3]) / 3 > 200:
                eachPix[0], eachPix[1], eachPix[2] = 255, 255, 255
            else:
                eachPix[0], eachPix[1], eachPix[2] = 0, 0, 0

    return newAr


def compareImage(target, shape):
    alike = 0
    h = len(target)
    w = len(target[0])
    for y in range(h):
        for x in range(w):
            if abs(sum(target[y][x][:3]) / 3 - sum(shape[y][x][:3]) / 3) / 255 < 0.5:
                alike += 1
            else:
                alike += -1

    return alike / (22 * 22)


def findcolor(target):
    h = len(target)
    w = len(target[0])
    # h,w=target.size #work if target is an image
    newar = np.zeros((h, w, 4))
    sumred = 0
    sumblue = 0
    sumgreen = 0
    for y in range((h // 2 - 1), h // 2 + 1):
        for x in range(w // 2 - 1, w // 2 + 1):
            sumred += target[y][x][0]
            sumblue += target[y][x][1]
            sumgreen += target[y][x][2]
    redchance = sumred / (sumred + sumblue + sumgreen)
    bluechance = sumblue / (sumred + sumblue + sumgreen)
    greenchance = sumgreen / (sumred + sumblue + sumgreen)
    # print('red='+str(redchance)+'blue='+str(bluechance)+'green='+str(greenchance))
    if redchance > 0.4 and bluechance > 0.3 and greenchance < 0.3:
        return 'Yellow'
    elif redchance < 0.5 and bluechance > 0.5 and greenchance > 0.2:
        return 'Green'
    elif redchance > 0.5 and bluechance < 0.5 and greenchance < 0.5:
        return 'Red'
    elif redchance > 0.3 and bluechance < 0.2 and greenchance > 0.5:
        return 'Purple'
    else:
        return 'Blue'


def findcolor2(target):
    h = len(target)
    w = len(target[0])
    red_min, red_max = (200, 0, 0), (255, 50, 50)  # RGB
    purple_min, purple_max = (200, 0, 200), (255, 50, 255)  #
    yellow_min, yellow_max = (200, 200, 0), (255, 255, 50)
    green_min, green_max = (0, 150, 0), (50, 255, 50)
    blue_min, blue_max = (0, 0, 150), (50, 50, 255)
    white_min, white_max = (200, 200.200), (200, 200, 200)
    colortype = ['Red', 'Purple', 'Yellow', 'Green', 'Blue','White']
    colorrange = [[red_min, red_max], [purple_min, purple_max], [yellow_min, yellow_max], [green_min, green_max],
                  [blue_min, blue_max], [white_min, white_max]]
    midpixel = target[h // 2][w // 2]
    for i in range(0, len(colorrange)):
        print(f'check {colortype[i]}:{colorrange[i][0][0]} >= {midpixel[0]} >= {colorrange[i][1][0]}:')
        if colorrange[i][0][0] <= midpixel[0] <= colorrange[i][1][0]:
            if colorrange[i][0][1] <= midpixel[1] <= colorrange[i][1][1]:
                if colorrange[i][0][2] <= midpixel[2] <= colorrange[i][1][2]:
                    print('this i ',i)

                    break
    #print('middlepixel=', midpixel)
    #print('color:', colortype[i])
    return colortype[i]


def findcolorshape(image):
    h, w = 22, 22
    testcirlce = Image.open('glycanshape/circleShape2.png')
    testsquare = Image.open('glycanshape/squareShape.png')
    testdiamond = Image.open('glycanshape/diamondShape2.png')
    testtriangle = Image.open('glycanshape/triangleShape.png')
    testtriangleup = Image.open('glycanshape/triangleupShape.png')
    testtriangleup = testtriangleup.resize((22, 22))
    testtriangle = testtriangle.resize((22, 22))
    testdiamond = testdiamond.resize((22, 22))
    testsquare = testsquare.resize((22, 22))
    testcirlce = testcirlce.resize((22, 22))

    i = image
    # print('file:', path)
    # i.show()
    i = i.resize((h, w))
    iarray = np.array(i)
    color = findcolor2(iarray)

    iarray = threshold(iarray)

    shape = ['Circle', 'Square', 'Diamond', 'Triangle', 'TriangleUp']
    shapechance = [compareImage(iarray, np.array(testcirlce)), compareImage(iarray, np.array(testsquare)),
                   compareImage(iarray, np.array(testdiamond)), compareImage(iarray, np.array(testtriangle)),
                   compareImage(iarray, np.array(testtriangleup))]
    i = shapechance.index(max(shapechance))
    # print(shape[i])
    return color, shape[i]

'''
image = Image.open('glycanshape/greencircleneedcheck.png')
# image.show()
color, shape = findcolorshape(image)
findcolor2(np.array(image))
# image = image.resize((22,22))
image = threshold(np.array(image))
image = Image.fromarray(image)
# image.save('glycanshape/circleShape2.png')
print(color, shape)
'''