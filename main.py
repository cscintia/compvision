import numpy as np
import cv2 as cv


def empty(x):
    pass


cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
cv.createTrackbar("Threshold1", "Parameters", 190, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 230, 255, empty)


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
def detect_shape(img, img_contour):
    # Using CHAIN_APPROX_NONE instead of CHAIN_APPROX_SIMPLE to get more contour points
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            cv.drawContours(img_contour, contours, -1, (217, 223, 29), 5)  # colour: vivid yellow
        perimeter = cv.arcLength(cnt, True)
        epsilon = 0.1 * perimeter
        approx = cv.approxPolyDP(cnt, epsilon, True)


    # Triangle
    if len(approx) == 3:
        shape = "triangle"

    # Square or rectangle
    if len(approx) == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)

    # A square will have an aspect ratio that is approximately
    # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

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


while True:
    img = cv.imread("/home/cintia/Desktop/SZE/3_felev/Gepi_latas/photos/Traffic-sign-road.jpg")

    # Before converting grayscale, we use blur function in order to reduce noise
    imgBlur = cv.GaussianBlur(img, (5, 5), 0)

    # Converting colour from RGB (Blur version) into gray
    imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

    imgContour = img.copy()

    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")

    # Detecting edges
    imgCanny = cv.Canny(imgGray, threshold1, threshold2)

    # Removing other noises
    kernel = np.ones((1, 1))
    imgDil = cv.dilate(imgCanny, kernel, iterations=1)

    img_contour = detect_shape(imgDil, imgContour)
    # Apply hough transform on the image

    gEdges = cv.Laplacian(imgGray, cv.CV_8UC1)
    circles = cv.HoughCircles(gEdges, cv.HOUGH_GRADIENT, 40, 10, param1=50,param2=60,minRadius=2,maxRadius=15)
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


    imgStack = stack_images(0.8, ([img, imgDil, imgContour]))

    # cv.imshow("Result", imgBlur)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #    break
    cv.imshow("Result", imgStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break