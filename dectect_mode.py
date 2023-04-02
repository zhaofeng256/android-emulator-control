import csv
import ctypes
import threading
import time
from operator import itemgetter

import cv2
import numpy as np
import pyautogui
from matplotlib import pyplot as plt

import tcp_service
from defs import TcpData, EventType, set_param1, set_param2, OFFSET_PARAM_1, OFFSET_PARAM_2, SupplyType, ControlEvent, \
    SubModeType, MapModeStatus, TransPointStatus, MainModeType
from window_info import window_info_init


def similar_point(a, b):
    return abs(a[0] - b[0]) < 3 and abs(a[1] - b[1]) < 3 and abs(a[2] - b[2]) < 4


def match_count(A, B):
    match = 0
    for a in A:
        for b in B:
            if similar_point(a, b):
                match += 1
                break

    return match


def get_match_item(a, M):
    match = []
    for i in range(len(M)):
        m = match_count(a, M[i])
        match.append(m)
        print('item', i, 'match', m)

    if max(match) == 0:
        return -1
    else:
        return np.argmax(match)


def detect_circles_by_capture():
    image = pyautogui.screenshot(region=[870, 350, 1280 - 870, 720 - 350])
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=38, maxRadius=45)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

    return circles


def detect_circles_of_file(name):
    image = cv2.imread(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=38, maxRadius=45)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

    return circles


def detect_boate():
    image = cv2.imread("boat.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=33, maxRadius=37)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

    return circles


def detect_sub_mode(img):
    moto = [[79, 190, 40], [79, 59, 41], [201, 55, 41]]
    chopper = [[118, 86, 42]]
    coyote = [[84, 263, 44], [156, 69, 40], [66, 143, 44], [290, 69, 43]]
    M = [moto, chopper, coyote]
    boat = [338, 299, 35]

    crop = img[350:720, 870:1280]
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=38, maxRadius=45)
    i = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        i = get_match_item(circles, M)
        print('is', i)
    else:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=33, maxRadius=37)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(circles)
            if len(circles) == 1:
                if similar_point(circles[0], boat):
                    i = 0
                    print('boat')
    return i


def sift_match_zone(img1, img2, rect, kp1, des1, sift, bf):
    show_img = False
    x1, y1, x2, y2 = rect
    x1 = round(float(x1)) - 2
    y1 = round(float(y1)) - 2
    x2 = round(float(x2)) + 2
    y2 = round(float(y2)) + 2
    crop = img2[y1:y2, x1:x2]

    # cv2.imshow("crop",crop)
    # cv2.waitKey(3)
    # print(x1,x2,y1,y2, crop.shape)

    kp2, des2 = sift.detectAndCompute(crop, None)
    if len(kp2) < 10:
        return False, 0, 0

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda val: val.distance)
    # print('matches num:', len(matches))
    # for m in matches:
    #     print(m.distance)

    # Draw first 5 matches need in Main Thread
    if show_img:
        out = cv2.drawMatches(img1, kp1, crop, kp2, matches[:5], None, flags=2)
        plt.imshow(out), plt.show()

        for i in range(5):
            print(matches[i].distance, x1 + kp2[matches[i].trainIdx].pt[0], y1 + kp2[matches[i].trainIdx].pt[1])

    if matches[0].distance < 30:
        print(matches[0].distance)
        return True, x1 + kp2[matches[0].trainIdx].pt[0], y1 + kp2[matches[0].trainIdx].pt[1]
    else:
        return False, 0, 0


def read_panel_axis():
    with open('panels.csv', 'r', newline='\n') as f:
        reader = csv.DictReader(f)
        # print(reader.line_num)
        axis = [*reader]
        # print(axis)
        # for row in reader:
        #     print(row.get('name'))
        f.close()
        axis = sorted(axis, key=lambda d: d['id'])
        return axis


def send_supply_position(type, id, p):
    data = TcpData()
    data.type = EventType.TYPE_LOCATION

    a = ctypes.c_int16(type)
    b = ctypes.c_int16(id)
    ctypes.memmove(ctypes.byref(data, OFFSET_PARAM_1), ctypes.byref(a), 2)
    ctypes.memmove(ctypes.byref(data, OFFSET_PARAM_1 + 2), ctypes.byref(b), 2)

    a = ctypes.c_int16(p[0])
    b = ctypes.c_int16(p[1])
    ctypes.memmove(ctypes.byref(data, OFFSET_PARAM_2), ctypes.byref(a), 2)
    ctypes.memmove(ctypes.byref(data, OFFSET_PARAM_2 + 2), ctypes.byref(b), 2)
    # tcp_service.tcp_data_append(data)


def detect_supply_box(img):
    show_img = False
    left = 368
    right = 877
    top = 286
    bottom = 571

    target_width = 216
    target_high = 61

    self_def_width = 248
    self_def_high = 176
    self_def_confirm = (693,596)
    crop = img[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    supply_type = SupplyType.SUPPLY_UNKOWN
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                if show_img:
                    crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                supply_type = SupplyType.SUPPLY_CUSTOM
                lst.append((x, y, w, h))
            elif abs(self_def_width - w) <= 3 and abs(self_def_high - h) <= 3:
                if show_img:
                    crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                supply_type = SupplyType.SUPPLY_SYSTEM
                print('custom supply', x, y, w, h)

    if supply_type == SupplyType.SUPPLY_SYSTEM:
        pos_custom_supply = [0, 0] * 4
        pos_custom_supply[0] = (left + round(0.5 * self_def_width), top + 20)
        pos_custom_supply[1] = (left + round(1.5 * self_def_width), top + 20)
        pos_custom_supply[2] = (left + round(0.5 * self_def_width), bottom - 20)
        pos_custom_supply[3] = (left + round(1.5 * self_def_width), bottom - 20)
        #select button position
        for i in range(4):
            send_supply_position(SupplyType.SUPPLY_SYSTEM, i, pos_custom_supply[i])
            if show_img:
                cv2.putText(img, str(i + 1), pos_custom_supply[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #confirm button position
        send_supply_position(SupplyType.MUX_BUTTON, 33, self_def_confirm)
        if show_img:
            cv2.putText(img, 'F', self_def_confirm, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif supply_type == SupplyType.SUPPLY_CUSTOM:
        lst = sorted(lst, key=itemgetter(0))
        lst = sorted(lst, key=itemgetter(1))
        for i in range(len(lst)):
            send_supply_position(SupplyType.SUPPLY_CUSTOM, i, (left + lst[i][0] + round(0.5 * target_width),
                                 top + lst[i][1] + round(0.5 * target_high)))
            if show_img:
                cv2.putText(img, str(i + 1),
                            (left + lst[i][0] + round(0.5 * target_width), top + lst[i][1] + round(0.5 * target_high)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print('system supply', i + 1, lst[i])

    if show_img and supply_type != SupplyType.SUPPLY_UNKOWN:
        win_name = "supply_box"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 0, 0)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def detect_random_supply(img):
    left = 681
    right = 933
    top = 283
    bottom = 571
    target_width = 216
    target_high = 60
    show_img = False

    crop = img[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    supply_type = SupplyType.SUPPLY_UNKOWN
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                if show_img:
                    crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                lst.append((x, y, w, h))
                supply_type = SupplyType.SUPPLY_RANDOM

    lst = sorted(lst, key=itemgetter(1))
    for i in range(len(lst)):
        send_supply_position(SupplyType.SUPPLY_RANDOM, i, (left + lst[i][0] + round(0.5 * target_width),
                             top + lst[i][1] + round(0.5 * target_high)))
        if show_img:
            cv2.putText(img, str(i + 1),
                        (left + lst[i][0] + round(0.5 * target_width), top + lst[i][1] + round(0.5 * target_high)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print('random supply', i + 1, lst[i])

    if show_img and supply_type != SupplyType.SUPPLY_UNKOWN:
        win_name = "supply_box"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 0, 0)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def detect_select_vehicle(img):
    left = 536
    right = 736
    top = 472
    bottom = 527
    target_width = 192
    target_high = 47
    pos_prev = (500, 500)
    pos_next = (775, 500)
    pos_confirm = (636, 542)
    show_img = True

    crop = img[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                if show_img:
                    cv2.rectangle(img, (left+x, top+y), (left+x+w, top+y+h), (0, 255, 0), 2)
                found = True
                print('select_vehicle found at', (left+x, top+y))

    if found and show_img:
        cv2.putText(img, 'Q', pos_prev, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, 'E', pos_next, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, 'F', pos_confirm, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        win_name = "select_vehicle"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 0, 0)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

class ModeInfo():
    def __init__(self):
        self.main_mode = MainModeType.MULTI_PLAYER
        self.sub_mode = SubModeType.NONE_SUB_MODE
        self.map_mode_on = False
        self.transparent_mode_on = False

mode_info = ModeInfo()

def main_mode_switch(mode):
    mode_info.main_mode = mode
    print("switch to mode", mode)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.MAIN_MODE)
    set_param2(data, int(mode))
    tcp_service.tcp_data_append(data)

def send_sub_mode(i):
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.SUB_MODE)
    set_param2(data, i)
    tcp_service.tcp_data_append(data)


def sub_mode_switch():

    if mode_info.sub_mode == SubModeType.NONE_SUB_MODE:
        i = detect_sub_mode()
        if i >= 0:
            mode_info.sub_mode = i + SubModeType.SUB_MODE_OFFSET
            print("switch to sub mode", mode_info.sub_mode)
            send_sub_mode(mode_info.sub_mode)
        else:
            print("no sub mode detected")
    else:
        mode_info.sub_mode = SubModeType.NONE_SUB_MODE
        send_sub_mode(mode_info.sub_mode)


def map_mode_switch():

    mode_info.map_mode_on = not mode_info.map_mode_on
    print("map mode on is", mode_info.map_mode_on)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.MAP_MODE)
    if mode_info.map_mode_on:
        set_param2(data, MapModeStatus.MAP_MODE_ON)
    else:
        set_param2(data, MapModeStatus.MAP_MODE_OFF)
    tcp_service.tcp_data_append(data)


def trans_point_mode_switch():
    window_info_init()
    mode_info.transparent_mode_on = not mode_info.transparent_mode_on
    print('transparent point mode on is', mode_info.transparent_mode_on)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.TRANSPARENT_MODE)
    if mode_info.transparent_mode_on:
        set_param2(data, TransPointStatus.TRANSPARENT_ON)
    else:
        set_param2(data, TransPointStatus.TRANSPARENT_OFF)
    tcp_service.tcp_data_append(data)



def detect_thread():
    axis = read_panel_axis()
    # print(axis)
    # TAKE_DRIVE = 0
    # DRIVE_BY = 1
    # DOOR = 2
    # ds = {'take_drive':TAKE_DRIVE, 'drive_by':DRIVE_BY, 'door':DOOR}

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    lsg_img = []
    lst_rect = []
    lst_kp = []
    lst_des = []

    for d in axis:
        name = d['name']
        img1 = cv2.imread('4/' + name + '.png', cv2.IMREAD_GRAYSCALE)
        lsg_img.append(img1)
        left, top, right, bottom = float(d['left']), float(d['top']), float(d['right']), float(d['bottom'])
        lst_rect.append((left, top, right, bottom))
        kp1, des1 = sift.detectAndCompute(img1, None)
        lst_kp.append(kp1)
        lst_des.append(des1)
        # print(name, 'keypoint num', len(kp1))


    bak_sub_mode = SubModeType.NONE_SUB_MODE
    while True:

        img2 = pyautogui.screenshot(region=(0, 0, 1280, 720))
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        n = len(axis)
        for i in range(n):
            found, x, y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift, bf)
            if found:
                xc = (lst_rect[i][0] + lst_rect[i][2]) / 2
                yc = (lst_rect[i][1] + lst_rect[i][3]) / 2
                print(axis[i]['name'], 'found at', (x, y), 'center offset is', (x - xc, y - yc))
                # send F point
                if axis[i]['name'] != 'drive_by':
                    send_supply_position(SupplyType.MUX_BUTTON, 33, (x, y))
                else:
                    send_supply_position(SupplyType.MUX_BUTTON, 34, (x, y))
            else:
                print(axis[i]['name'], 'not found')

        detect_supply_box(img2)
        detect_random_supply(img2)
        detect_select_vehicle(img2)


        mod = detect_sub_mode(gray)
        if bak_sub_mode != mod:
            print("switch to sub mode",mod)
            send_sub_mode(mod+SubModeType.SUB_MODE_OFFSET)
            bak_sub_mode = mod

        time.sleep(1)


def detect_service():
    t = threading.Thread(target=detect_thread, args=())
    t.daemon = True
    t.start()
    return t


def show_full_screen(name):
    image = cv2.imread(name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey(5)


def ll():
    time.sleep(1)
    detect_thread()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_full_screen('3/coyote.png')
    ll()
    # show_full_screen('3/select_vehicle.png')
    # ll()
    # show_full_screen('3/print_vehicle_1.png')
    # ll()
    # show_full_screen('3/tough_on.png')
    # ll()
    # show_full_screen('3/random_supply.png')
    # ll()
    # show_full_screen('3/custom_supply.png')
    # ll()
    # show_full_screen('3/system_supply.png')
    # ll()
    # show_full_screen('3/system_supply_1.png')
    # ll()
    # t = detect_service()
    # t.join()
    #
    # show_full_screen('3/take_drive.png')
    # ll()
    #
    # show_full_screen('3/door.png')
    # ll()
    #
    # show_full_screen('3/strop_on.png')
    # ll()
    #
    # show_full_screen('3/strop_off.png')
    # ll()
    # img1 = cv2.imread('4/' + name + '.png')
    # img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
