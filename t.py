import cv2
import numpy as np
import sys
import statistics
import time

# Reading Video
cap = cv2.VideoCapture('1.mp4')


while cap.isOpened():
    # time.sleep(.125)
    ret, frame = cap.read()
    if not ret:
        break
    frame1 = frame

    a, b, c = frame.shape


    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
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
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])

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

    # for fc in frcontours:
    #     if cv2.contourArea(fc) > 250:
    #         cv2.drawContours(frame, fc, -1, (0, 0, 0), 2)
    for fc in focontours:
        if cv2.contourArea(fc)  >150 and cv2.contourArea(fc) < 1000:
            cv2.drawContours(frame, fc, -1, (0, 0, 0), 2)
    # for fc in fbcontours:
    #     if cv2.contourArea(fc) > 250:
    #         cv2.drawContours(frame, fc, -1, (0, 0, 0), 2)
    # for fc in fblcontours:
    #     if cv2.contourArea(fc) > 250:
    #         cv2.drawContours(roi, fc, -1, (0, 0, 0), 2)
    # for fc in fgcontours:
    #     # if cv2.contourArea(fc) > 250:
    #     cv2.drawContours(frame, fc, -1, (0, 0, 0), 2)

    cv2.imshow('result', frame)
    # cv2.imshow('result', frame1)
    # cv2.imshow('result', roi2)
    k = cv2.waitKey(15) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()