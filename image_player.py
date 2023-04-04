import os
import time

import cv2
import keyboard

idx = 0
lst_pic = os.listdir('5')
t = len(lst_pic)
def key_event_callback(evt):
    global idx
    if evt.event_type == 'down':
        if evt.scan_code == 72:
            idx -= 1
            show_full_screen(lst_pic[idx%t])
            pass
        elif evt.scan_code == 80:
            idx += 1
            show_full_screen(lst_pic[idx%t])
            pass
    #print(evt.name, evt.scan_code, evt.event_type)

def show_full_screen(name):#
    print(name)
    cv2.destroyAllWindows()
    image = cv2.imread('5/' + name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey()


if __name__ == '__main__':
    keyboard.hook(key_event_callback)
    while True:
        time.sleep(1)