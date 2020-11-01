import cv2
import numpy as np
import sys
import statistics
import time

# Reading Video
cap = cv2.VideoCapture('1.mp4')

# Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
# Making matrix for Erosion, dilation and morphing
kernel = np.ones((2, 2), np.uint8)
kernel1 = np.ones((1, 2), np.uint8)

#Talha changes 
# Shary's changes

# (major_ver, minor_ver, subminor_ver) = (cv2._version_).split('.')
# if int(major_ver) < 3:
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: {0}".format(fps))
# else:
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print("FPS: {0}".format(fps))

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
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        elif cv2.contourArea(c) > 1000:
            # print(len(contours))
            (x, y, w, h) = cv2.boundingRect(c)

            if x < b / 2:
                # Using cv2.putText() method
                # frame = cv2.putText(frame, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                leftwidth.append(w)
                leftheight.append(h)
            elif x > b / 2:
                # frame = cv2.putText(frame, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                rightwidth.append(w)
                rightheight.append(h)
cap.release()


if b in leftwidth :
    leftwidth.remove(b)
if a in leftheight :
    leftheight.remove(a)

lw = statistics.mode(leftwidth)
lh = statistics.mode(leftheight)

if b in rightwidth:
    rightwidth.remove(b)
if a in rightwidth:
    rightheight.remove(a)

rw = statistics.mode(rightwidth)
rh = statistics.mode(rightheight)

avgw = (lw+rw)/2
avgh = (lh+rh)/2

print(int(avgw),int(avgh))
# t = int(time.time())*1000
# print(time.time()*1000)
# Going Through Video again


# Reading Video
cap = cv2.VideoCapture('1.mp4')

# Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
# Making matrix for Erosion, dilation and morphing
kernel = np.ones((2, 2), np.uint8)
kernel1 = np.ones((1, 2), np.uint8)

it = 0
countL = 0
countR = 0
pLx=0
pLy=0
pRx=0
pRy=0
pLw=0
pRw=0
pLh=0
pRh=0
p1 = 0.3
p2 = 0.7


while cap.isOpened():
    # time.sleep(.125)
    ret, frame = cap.read()
    if not ret:
        break
    frame1 = frame

    a, b, c = frame.shape
    # print(a,b)

    mask1 = mask.apply(frame)
    # Erosion
    mask1 = cv2.erode(mask1, kernel, iterations=1)
    # Dialtion
    mask1 = cv2.dilate(mask1, kernel1, iterations=1)
    # Morphing
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contFound = False
    for c in contours:
        if cv2.contourArea(c) < 400:
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
                # Using cv2.putText() method
                # pLx = p1 * pLx
                # x = p2 * x
                # pLy = p1 * pLy
                # y = p2 * y
                # x = int((pLx + x))
                # y = int((pLy + y))
                frame = cv2.putText(frame, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                w = lw
                h = lh

                countL += 1
                pLx = x
                pLy = y
                pLw = w
                pLh = h
                # if abs(oldlx-x) > 25:
                #     x = oldlx
                # oldlx = x

            elif x > b / 2 and countR < 1:
                # Using cv2.putText() method
                # pRx = p1 * pRx
                # x = p2 * x
                # pRy = p1 * pRy
                # y = p2 * y
                # x = int((pRx + x))
                # y = int((pRy + y))
                frame = cv2.putText(frame, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                w = rw
                h = rh
                countR += 1
                pRx = x
                pRy = y
                pRw = w
                pRh = h

                # if abs(oldrx-x) > 25:
                # x = oldrx
                # oldrx = x

            # cv2.rectangle(frame, (pLx, pLy), (pLx + pLw, pLy + pLh), (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)


            upper_left = (x, y)
            bottom_right = (x + w, y + h)
            roi = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

            cw = w / 2
            ch = h / 2
            cv2.circle(roi, (int(cw), int(ch)), 3, (0, 0, 0), -1)

            blurred_frame = cv2.GaussianBlur(roi, (5, 5), 0)
            # hsv1 = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)

            # define range of red color in HSV
            lower_red = np.array([110, 50, 50])
            upper_red = np.array([130, 255, 255])
            # define range of orange color in HSV
            lower_orange = np.array([91, 50, 50])
            upper_orange = np.array([111, 255, 255])
            # define range of brown color in HSV
            lower_brown = np.array([102, 50, 50])
            upper_brown = np.array([122, 255, 255])
            # define range of green color in HSV
            lower_green = np.array([50, 100, 100])
            upper_green = np.array([70, 255, 255])
            # define range of blue color in HSV
            lower_blue = np.array([154, 2, 21])
            upper_blue = np.array([174, 255, 255])

            # Threshold the HSV image to get only blue colors
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            frcontours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            focontours, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            fbcontours, _ = cv2.findContours(mask_brown, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            fblcontours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            fgcontours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for fc in frcontours:
                if cv2.contourArea(fc) > 400:
                    # time.sleep(.8)
                    # cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)
                    (fx,fy,fw,fh) = cv2.boundingRect(fc)
                    # cv2.rectangle(roi, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 2)

                    upper_left = (fx, fy)
                    bottom_right = (fx + fw, fy + fh)
                    roi1 = roi[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

                    fw = fw/2
                    fh = fh/2
                    cv2.circle(roi1, (int(fw), int(fh)), 3, (255, 255, 255), -1)
            # for fc in focontours:
            #     if cv2.contourArea(fc) > 250:
            #         cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)
            # for fc in fbcontours:
            #     if cv2.contourArea(fc) > 250:
            #         cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)
            # for fc in fblcontours:
            #     if cv2.contourArea(fc) > 250:
            #         cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)
            # for fc in fgcontours:
            #     if cv2.contourArea(fc) > 250:
            #         cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)

    # cv2.imshow('result', masked_image)
    if contFound == False:
        cv2.rectangle(frame, (pLx, pLy), (pLx + pLw, pLy + pLh), (255, 255, 255), 2)
        cv2.rectangle(frame, (pRx, pRy), (pRx + pRw, pRy + pRh), (255, 255, 255), 2)


    cv2.imshow('result', frame)
    # cv2.imshow('result', frame1)
    # cv2.imshow('result', roi2)
    k = cv2.waitKey(15) & 0xFF
    if k == 27:
        break
    countL = 0
    countR = 0
    it += 1
# This is where the video is read

cap.release()
cv2.destroyAllWindows()
# print(time.time()*1000)
#
# while 1>0:
#     black_bg = 255 * np.ones_like(frame)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
#     cv2.imshow('sec', frame)
# cv2.destroyAllWindows()