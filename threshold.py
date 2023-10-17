from __future__ import print_function

import cv2

import threading

cd = threading.Condition()
window_origin_name = 'origin'
window_trackbar_name = 'trackbar'
window_thresh_name = 'thresh'
low_H_name = 'LowH'
low_S_name = 'LowS'
low_V_name = 'LowV'
high_H_name = 'HighH'
high_S_name = 'HighS'
high_V_name = 'HighV'

low_h, low_s, low_v = 0, 0, 0
high_h, high_s, high_v = 255, 255, 255

frame = cv2.imread('3/select_vehicle.png')
frame = frame[514:570,526:746]
# frame = cv2.imread('3/armor_off.png')
# frame = frame[456:514, 580:700]
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def update():
    frame_threshold = cv2.inRange(frame_hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
    cv2.imshow(window_thresh_name, frame_threshold)


def on_trackbar_low_h(v):
    global low_h
    low_h = v
    update()

def on_trackbar_low_s(v):
    global low_s
    low_s = v
    update()

def on_trackbar_low_v(v):
    global low_v
    low_v = v
    update()

def on_trackbar_high_h(v):
    global high_h
    high_h = v
    update()

def on_trackbar_high_s(v):
    global high_s
    high_s = v
    update()

def on_trackbar_high_v(v):
    global high_v
    high_v = v
    update()

def main():

    cv2.namedWindow(window_trackbar_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_trackbar_name, 600, 300)
    cv2.createTrackbar(low_H_name, window_trackbar_name, 0, 255, on_trackbar_low_h)
    cv2.createTrackbar(high_H_name, window_trackbar_name, 255, 255, on_trackbar_high_h)
    cv2.createTrackbar(low_S_name, window_trackbar_name, 0, 255, on_trackbar_low_s)
    cv2.createTrackbar(high_S_name, window_trackbar_name, 255, 255, on_trackbar_high_s)
    cv2.createTrackbar(low_V_name, window_trackbar_name, 0, 255, on_trackbar_low_v)
    cv2.createTrackbar(high_V_name, window_trackbar_name, 255, 255, on_trackbar_high_v)

    cv2.imshow(window_origin_name, frame)

    cv2.waitKey()


if __name__ == '__main__':
    main()
