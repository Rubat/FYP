import cv2
import numpy as np
import argparse
import statistics
from collections import deque

pXRDot = []
pYRDot = []

pXODot = []
pYODot = []

pXBDot = []
pYBDot = []

pXBLDot = []
pYBLDot = []

pXGDot = []
pYGDot = []


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])


def Dots(fcontours, x, y, color):
    for fc in fcontours:
        if cv2.contourArea(fc) > 400:
            (fx, fy, fw, fh) = cv2.boundingRect(fc)

            if color == 'red':
                pXRDot.append(fx + x)
                pYRDot.append(fy + y)
            if color == 'orange':
                pXODot.append(fx + x)
                pYODot.append(fy + y)
            if color == 'brown':
                pXBDot.append(fx + x)
                pYBDot.append(fy + y)
            if color == 'blue':
                pXBLDot.append(fx + x)
                pYBLDot.append(fy + y)
            if color == 'green':
                pXGDot.append(fx + x)
                pYGDot.append(fy + y)
    for i in range(0, len(pYBLDot)):
        if pYBLDot[i] > 160:
            pYBLDot[i] = pYBLDot[i-1]


# Reading Video
vidname = 'new1.mp4'
cap = cv2.VideoCapture(vidname)

# Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
# Making matrix for Erosion, dilation and morphing
kernel = np.ones((2, 2), np.uint8)
kernel1 = np.ones((2, 2), np.uint8)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: {0}".format(fps))

# Global Variables
leftwidth = []
rightwidth = []
leftheight = []
rightheight = []
oldlx = 0
oldrx = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    frame1 = frame

    a, b, c = frame.shape


    mask1 = mask.apply(frame)
    # Erosion
    mask1 = cv2.erode(mask1, kernel, iterations=1)
    # Dialtion
    mask1 = cv2.dilate(mask1, kernel1, iterations=3)
    # Morphing
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        elif cv2.contourArea(c) > 1000:

            (x, y, w, h) = cv2.boundingRect(c)

            if x < b / 2:

                leftwidth.append(w)
                leftheight.append(h)
            elif x > b / 2:

                rightwidth.append(w)
                rightheight.append(h)
cap.release()

if b in leftwidth:
    leftwidth.remove(b)
if a in leftheight:
    leftheight.remove(a)

lw = statistics.mode(leftwidth)
lh = statistics.mode(leftheight)

if b in rightwidth:
    rightwidth.remove(b)
if a in rightwidth:
    rightheight.remove(a)

rw = statistics.mode(rightwidth)
rh = statistics.mode(rightheight)

avgw = (lw + rw) / 2
avgh = (lh + rh) / 2

# Reading Video

cap = cv2.VideoCapture(vidname)

# Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
# Making matrix for Erosion, dilation and morphing
kernel = np.ones((2, 2), np.uint8)
kernel1 = np.ones((2, 2), np.uint8)

it = 0
countL = 0
countR = 0
pLx = 0
pLy = 0
pRx = 0
pRy = 0
pLw = 0
pRw = 0
pLh = 0
pRh = 0
p1 = 0.5
p2 = 0.5

while cap.isOpened():
    # time.sleep(0.25)
    ret, frame = cap.read()
    if not ret:
        break
    frame1 = frame

    a, b, c = frame.shape
    # print(a,b)

    mask1 = mask.apply(frame)
    # Erosion
    mask1 = cv2.erode(mask1, kernel, iterations=1)
    # Dilation
    mask1 = cv2.dilate(mask1, kernel1, iterations=1)
    # Morphing

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contFound = False
    for c in contours:
        if cv2.contourArea(c) < 500:
            contFound = False
            continue
        elif cv2.contourArea(c) > 3500:
            # print(len(contours))
            contFound = True
            (x, y, w, h) = cv2.boundingRect(c)
            # print(cv2.contourArea(c))
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (x, y)

            # fontScale
            fontScale = 0.75

            # Blue color in BGR
            color = (255, 255, 100)

            # Line thickness of 2 px
            thickness = 2
            text = str(x) + ", " + str(y)

            if (x < b / 2 and countL < 1):

                frame = cv2.putText(frame, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                w = lw
                h = lh

                countL += 1
                pLx = x
                pLy = y
                pLw = w
                pLh = h

                cv2.rectangle(frame, (pLx, pLy), (pLx + pLw, pLy + pLh), (255, 255, 255), 2)

                upper_left = (pLx, pLy)
                bottom_right = (pLx + pLw, pLy + pLh)
                roi = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

                cw = w / 2
                ch = h / 2

                blurred_frame = cv2.GaussianBlur(roi, (5, 5), 0)

                hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)

                # define range of blue color in HSV
                lower_blue = np.array([110, 100, 10])
                upper_blue = np.array([130, 255, 255])

                # Threshold the HSV image to get only blue colors
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

                fblcontours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                Dots(fblcontours, x,y, 'blue')
                for i in range(1, len(pXBLDot)):
                    cv2.circle(frame, (int(pXBLDot[i]), int(pYBLDot[i])), 3, (255, 0, 0, 0), -1)
                    thickness = 1
                    cv2.line(frame, (pXBDot[i - 1], pYBDot[i - 1]), (pXBDot[i], pYBDot[i]), (0, 255, 255), thickness)

            elif x > b / 2 and countR < 1:

                frame = cv2.putText(frame, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                w = rw
                h = rh
                countR += 1
                pRx = x
                pRy = y
                pRw = w
                pRh = h

    if contFound == False:
        cv2.rectangle(frame, (pLx, pLy), (pLx + pLw, pLy + pLh), (255, 255, 255), 2)
        cv2.rectangle(frame, (pRx, pRy), (pRx + pRw, pRy + pRh), (255, 255, 255), 2)

    cv2.imshow('result', frame)
    cv2.imshow('result', frame1)
    k = cv2.waitKey(15) & 0xFF
    if k == 27:
        break
    countL = 0
    countR = 0
    it += 1

print(len(pXBLDot))
print(len(pYBLDot))
cap.release()
cv2.destroyAllWindows()