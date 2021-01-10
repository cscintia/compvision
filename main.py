import numpy as np
import cv2 as cv
import random as rng
import tkinter as tk
from tkinter import filedialog
rng.seed(12345)


# Stacking images together
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                img_array[0][0].shape[1], img_array[0][0].shape[0], None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]),
                                         None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


# Detect shapes
def detect_shape(img_dil, img_contour, img_orig):
    # Using CHAIN_APPROX_NONE instead of CHAIN_APPROX_SIMPLE to get more contour points
    contours, hierarchy = cv.findContours(img_dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    largest_area = 0
    best_cnt = contours[0]
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            best_cnt = cnt

    cv.drawContours(img_contour, best_cnt, -1, (29, 223, 217), 5)  # colour: vivid yellow
    
    perimeter = cv.arcLength(best_cnt, True)
    old_value = perimeter
    old_min = 0
    new_max = 100
    new_min = 3
    old_max = 10000
    epsilon = (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    print(epsilon)
    approx = cv.approxPolyDP(best_cnt, epsilon, True)

    shape = "Undetected"
    if len(approx) >= 3 and len(approx) <= 10:

        # Triangle
        if len(approx) == 3:
            shape = "triangle"

        # Rectangle
        if len(approx) == 4:
            shape = "rectangle"

        # Pentagon
        if len(approx) == 5:
            shape = "pentagon"

        # Hexagon
        if len(approx) == 6:
            shape = "hexagon"

        # Octagon
        if len(approx) == 8:
            shape = "octagon"

        # Star
        if len(approx) == 10:
            shape = "star"

    # Otherwise assume as circle or oval
    else:
        shape = "circle"
    print(shape)

    white = 0
    yellow = 0
    red = 0
    blue = 0

    # Detecting colour within the bounding rectangle in a specific line
    x, y, w, h = cv.boundingRect(best_cnt)
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.rectangle(img_contour, (x, y, w, h), color, 2)
    for i in range(y, y+h):
        # Detecting the colour of each pixel in a specific line
        if (img_orig[i, round(x+w/2)][0] >= 175 and img_orig[i, round(x+w/2)][0] <= 255 and
                img_orig[i, round(x+w/2)][1] >= 175 and img_orig[i, round(x+w/2)][1] <= 255 and
                img_orig[i, round(x+w/2)][2] >= 175 and img_orig[i, round(x+w/2)][2] <= 255):
            white += 1
        if (img_orig[i, round(x+w/2)][2] >= 150 and img_orig[i, round(x+w/2)][2] <= 255 and
                img_orig[i, round(x+w/2)][1] >= 0 and img_orig[i, round(x+w/2)][1] <= 100 and
                img_orig[i, round(x+w/2)][0] >= 0 and img_orig[i, round(x+w/2)][0] <= 100):
            red += 1
        if (img_orig[i, round(x + w / 2)][0] >= 175 and img_orig[i, round(x + w / 2)][0] <= 255 and
                img_orig[i, round(x + w / 2)][2] >= 0 and img_orig[i, round(x + w / 2)][2] <= 100 and
                img_orig[i, round(x + w / 2)][1] >= 0 and img_orig[i, round(x + w / 2)][1] <= 125):
            blue += 1
        if (img_orig[i, round(x + w / 2)][0] >= 0 and img_orig[i, round(x + w / 2)][0] <= 100 and
                img_orig[i, round(x + w / 2)][1] >= 125 and img_orig[i, round(x + w / 2)][1] <= 255 and
                img_orig[i, round(x + w / 2)][2] >= 200 and img_orig[i, round(x + w / 2)][2] <= 255):
            yellow += 1
    if shape != "triangle" and shape != "circle" and shape != "octagon":
        for i in range(x, x+w):
            # Detecting the colour of each pixel in a specific line
            if (img_orig[round(y+h/2), i][0] >= 175 and img_orig[round(y+h/2), i][0] <= 255 and
                    img_orig[round(y+h/2), i][1] >= 175 and img_orig[round(y+h/2), i][1] <= 255 and
                    img_orig[round(y+h/2), i][2] >= 175 and img_orig[round(y+h/2), i][2] <= 255):
                white += 1
            if (img_orig[round(y+h/2), i][2] >= 150 and img_orig[round(y+h/2), i][2] <= 255 and
                    img_orig[round(y+h/2), i][1] >= 0 and img_orig[round(y+h/2), i][1] <= 100 and
                    img_orig[round(y+h/2), i][0] >= 0 and img_orig[round(y+h/2), i][0] <= 100):
                red += 1
            if (img_orig[round(y+h/2), i][0] >= 175 and img_orig[round(y+h/2), i][0] <= 255 and
                    img_orig[round(y+h/2), i][2] >= 0 and img_orig[round(y+h/2), i][2] <= 100 and
                    img_orig[round(y+h/2), i][1] >= 0 and img_orig[round(y+h/2), i][1] <= 125):
                blue += 1
            if (img_orig[round(y+h/2), i][0] >= 0 and img_orig[round(y+h/2), i][0] <= 100 and
                    img_orig[round(y+h/2), i][1] >= 125 and img_orig[round(y+h/2), i][1] <= 255 and
                    img_orig[round(y+h/2), i][2] >= 200 and img_orig[round(y+h/2), i][2] <= 255):
                yellow += 1

    print(white, red, blue, yellow)

    road_sign = "Undetected"
    if shape == "circle":
        if white > blue and white > yellow and red > white and white > 0 and red >= ((white+red+blue+yellow)/2):
            road_sign = "No Entry"
    if shape == "octagon":
        if red > blue and red > yellow and red > white and red >= ((white+red+blue+yellow)/1.5):
            road_sign = "Stop"
    if shape == "rectangle":
        if white > blue and white > red and yellow > white and white > 0 and yellow >= ((white+red+blue+yellow)/3):
            road_sign = "Main Road"
        if (white > yellow and red > yellow and blue > red and
                white > 0 and red > 0 and blue >= ((white+red+blue+yellow)/4)):
            road_sign = "No Through"
    if shape == "triangle":
        if red > blue and red > yellow and white > red and red > 0 and white >= ((white+red+blue+yellow)/3):
            road_sign = "Yield"
    print(road_sign)


root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

img = cv.imread(file_path)

# Before converting grayscale, we use bilateralFilter function in order to reduce noise
dst = cv.bilateralFilter(img, 10, 60, 60)

# Converting colour from RGB (Blur version) into gray
imgGray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

imgContour = img.copy()

threshold1 = 190
threshold2 = 230

# Detecting edges
imgCanny = cv.Canny(imgGray, threshold1, threshold2)

# Removing other noises
kernel = np.ones((1, 1))
imgDil = cv.dilate(imgCanny, kernel, iterations=1)

detect_shape(imgDil, imgContour, img)

imgStack = stack_images(0.8, ([img, imgCanny, imgContour]))

cv.imshow("Result", imgStack)
while 1:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
