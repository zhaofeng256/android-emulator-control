import os
import threading
import time

import cv2
import keyboard
folder='3'
idx = 0
lst_pic = os.listdir(folder)
t = len(lst_pic)
exit_app = False
c = threading.Condition()
dc = {}


def key_event_callback(evt):
    global idx, exit_app, dc

    if evt.name in dc.keys():
        if dc[evt.name] == evt.event_type:
            return
    print(evt.scan_code, evt.event_type)
    dc[evt.name] = evt.event_type

    if evt.event_type == 'down':
        if evt.scan_code == 72:
            with c:
                idx -= 1
                print('call', idx)


        elif evt.scan_code == 80:
            with c:
                idx += 1
                print('call', idx)


        elif evt.scan_code == 66:
            exit_app = True


def show_full_screen(file_name):  #
    global folder
    image = cv2.imread(folder +'/' + file_name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey()


if __name__ == '__main__':
    keyboard.hook(key_event_callback, suppress=False)

    bak_name = ''
    while not exit_app:
        time.sleep(0.1)
        with c:
            name = lst_pic[idx % t]
        if name == bak_name:
            continue
        print(idx, name)
        show_full_screen(name)

# lst_pic = ['print_vehicle.png','select_vehicle.png','select_weapon.png',
#            'tough_on.png','tough_off.png','update_chip.png', 'moto.png',
#            'take_drive.png', 'random_supply.png','system_supply.png',
#            'custom_supply.png','tough_on.png','door.png']
