import csv
import ctypes
import os
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
        #print('item', i, 'match', m)

    if max(match) == 0:
        return -1
    else:
        return np.argmax(match)


def detect_drive_mode(img):
    moto = cnvt_circles([[79, 190, 40], [79, 59, 41], [201, 55, 41]])
    chopper = cnvt_circles([[118, 86, 42]])
    coyote = cnvt_circles([[84, 263, 44], [156, 69, 40], [66, 143, 44], [290, 69, 43]])
    M = [moto, chopper, coyote]
    boat = cnvt_circles([[338, 299, 35]])
    dc = {SubModeType.NONE_SUB_MODE:'none',SubModeType.DRIVE_MOTO:'moto',
          SubModeType.DRIVE_CHOPPER:'chopper',SubModeType.DRIVE_COYOTE:'coyote'}

    crop = img[round(cnvt_x(350)):wd_high, round(cnvt_x(870)):wd_width]
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=int(cnvt_x(38)), maxRadius=int(cnvt_x(45)))
    i = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        i = get_match_item(circles, M)
        #print('is', i)
    else:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                                   param1=60, param2=40, minRadius=33, maxRadius=37)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(circles)
            if len(circles) == 1:
                if similar_point(circles[0], boat[0]):
                    i = 0

    i += SubModeType.SUB_MODE_OFFSET         #print('boat')
    return i, dc[i]


def send_sub_mode(mod):
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.SUB_MODE)
    set_param2(data, mod)
    tcp_service.tcp_data_append(data)


def sift_match_zone(img1, img2, rect, kp1, des1, sift, bf):
    show_img = False
    x1, y1, x2, y2 = rect
    x1 = round(float(x1)) - 2
    y1 = round(float(y1)) - 2
    x2 = round(float(x2)) + 2
    y2 = round(float(y2)) + 2
    crop = img2[y1:y2, x1:x2]

    kp2, des2 = sift.detectAndCompute(crop, None)
    if len(kp2) < 10:
        return False, 0, 0

    #matches = bf.match(des1, des2)
    #matches = sorted(matches, key=lambda val: val.distance)

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    rs = []
    for m, n in matches:
        ratio = m.distance / n.distance
        if ratio < 0.3:
            good.append(m)
            rs.append(ratio)

    min_r = 1
    n_good = len(good)
    if n_good > 0:
        min_r = min(rs)
    # Draw first 5 matches need in Main Thread
    if show_img:
        #out = cv2.drawMatches(img1, kp1, crop, kp2, matches[:5], None, flags=2)
        if n_good > 2:
            print('good', n_good, 'best', min_r)
            for i in range(n_good):
                print(rs[i], good[i].distance, x1 + kp2[good[i].trainIdx].pt[0], y1 + kp2[good[i].trainIdx].pt[1])

            out = cv2.drawMatches(img1, kp1, crop, kp2, good, None, flags=2)
            imS = cv2.resize(out, (800, 400))
            cv2.imshow("match", imS)
            cv2.waitKey()

        # for i in range(5):
        #     print(matches[i].distance, x1 + kp2[matches[i].trainIdx].pt[0], y1 + kp2[matches[i].trainIdx].pt[1])

    # if matches[0].distance < 60:
    #     print('match distance',matches[0].distance)
    #     return True, x1 + kp2[matches[0].trainIdx].pt[0], y1 + kp2[matches[0].trainIdx].pt[1]
    # else:
    #     return False, 0, 0

    if n_good > 0:
        best = good[np.argmin(rs)]
        return True, x1 + kp2[best.trainIdx].pt[0], y1 + kp2[best.trainIdx].pt[1]
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
    print('send key', id, 'at', p)
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
    # custom and system supply religion
    left, top, right, bottom = [int(i) for i in cnvt_rect(368, 286, 877, 571)]
    # system supply size
    target_width, target_high = cnvt_size(216, 61)
    # custom supply size
    self_def_width, self_def_high = cnvt_size(248, 176)

    crop = img[top:bottom, left:right]
    ret, thresh = cv2.threshold(crop, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    supply_type = SupplyType.SUPPLY_UNKOWN
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                supply_type = SupplyType.SUPPLY_SYSTEM
                lst.append((left + x + 0.5 * w, top + y + 0.5 * h, w, h))
            elif abs(self_def_width - w) <= 3 and abs(self_def_high - h) <= 3:
                supply_type = SupplyType.SUPPLY_CUSTOM
                lst = [(left + 0.5 * self_def_width, top + 20, w, 20),
                       (left + 1.5 * self_def_width, top + 20, w, 20),
                       (left + 0.5 * self_def_width, bottom - 20, w, 20),
                       (left + 1.5 * self_def_width, bottom - 20, w, 20)]
                break

    if supply_type == SupplyType.SUPPLY_SYSTEM:
        lst = sorted(lst, key=itemgetter(0))
        lst = sorted(lst, key=itemgetter(1))

    return supply_type, lst


def detect_supply(img):
    # random supply religion and size
    left, top, right, bottom = [int(i) for i in cnvt_rect(681, 283, 933, 571)]
    target_width, target_high = cnvt_size(216, 60)

    crop = img[top:bottom, left:right]
    ret, thresh = cv2.threshold(crop, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    supply_type = SupplyType.SUPPLY_UNKOWN
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                lst.append((left + x + 0.5 * w, top + y + 0.5 * h, w, h))
                supply_type = SupplyType.SUPPLY_RANDOM

    if supply_type == SupplyType.SUPPLY_UNKOWN:
        supply_type, lst = detect_supply_box(img)
    else:
        lst = sorted(lst, key=itemgetter(1))

    return supply_type, lst


def detect_select_vehicle(img):
    left, top, right, bottom = [int(i) for i in cnvt_rect(536, 472, 736, 527)]
    target_width, target_high = cnvt_size(192, 47)

    crop = img[top:bottom, left:right]
    ret, thresh = cv2.threshold(crop, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    a, b = (0, 0), (0, 0)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                found = True
                print('select_vehicle found at', (left + x, top + y))
                a, b = ((left + x, top + y), (left + x + w, top + y + h))
                break

    return found, a, b


mc_width = 1280
mc_height = 720

wd_left = 32
wd_top = 38
wd_width = 1213
wd_high = 682


def cnvt_x(x):
    return x * wd_width / mc_width


def cnvt_y(y):
    return y * wd_high / mc_height


def cnvt_pos(p):
    return round(cnvt_x(p[0])), round(cnvt_y(p[1]))



def cnvt_region(x, y, w, h):
    return wd_left + cnvt_x(x), wd_top + cnvt_y(y), cnvt_x(w), cnvt_y(h)


def cnvt_rect(left, top, right, bottom):
    return cnvt_x(left), cnvt_y(top), cnvt_x(right), cnvt_y(bottom)


def cnvt_size(width, high):
    return cnvt_x(width), cnvt_y(high)


def cnvt_circles(circles):
    return [[cnvt_x(c[0]), cnvt_y(c[1]), cnvt_x(c[2])] for c in circles]


def revt_x(x):
    return round(x * mc_width / wd_width)


def revt_y(y):
    return round(y * mc_height / wd_high)


def recvt_pos(x, y):
    return round(revt_x(x)), round(revt_y(y))


class DetectModeService():
    show_img = True
    self_def_confirm = (693, 596) # red supply enter
    pos_prev = (500, 500) # prev vehicle
    pos_next = (775, 500) # next vehicle
    pos_confirm = (636, 542) # vehicle confirm
    pos_auto_pick = (913, 218) # auto pick
    bak_f_x, bak_f_y = (0, 0)
    bak_g_x, bak_g_y = (0, 0)
    bak_ex_x, bak_ex_y = (0, 0)

    bak_t_s = SupplyType.SUPPLY_UNKOWN
    bak_lst_s = []
    bak_sel_vel = False

    def detect_thread(self):
        axis = read_panel_axis()
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
            lst_rect.append(cnvt_rect(left, top, right, bottom))
            kp1, des1 = sift.detectAndCompute(img1, None)
            lst_kp.append(kp1)
            lst_des.append(des1)
            # print(name, 'keypoint num', len(kp1))

        n = len(axis)
        bak_sub_mode = SubModeType.NONE_SUB_MODE

        idx = 0
        while True:
            r =[int(i) for i in cnvt_region(0, 0, mc_width, mc_height)]
            img = pyautogui.screenshot(region=r)
            cv2.destroyAllWindows()
            img2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 1. find F
            f_x, f_y = DetectModeService.pos_auto_pick
            for i in range(n):
                if int(axis[i]['key_code']) == 33:
                    found, f_x, f_y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift, bf)
                    if found:
                        xc = (lst_rect[i][0] + lst_rect[i][2]) / 2
                        yc = (lst_rect[i][1] + lst_rect[i][3]) / 2
                        print(axis[i]['name'], 'found at', recvt_pos(f_x, f_y), 'center offset is',
                              (round(f_x - xc), round(f_y - yc)))
                        break

            if f_x != DetectModeService.bak_f_x or f_y != DetectModeService.bak_f_y:
                DetectModeService.bak_f_x = f_x
                DetectModeService.bak_f_y = f_y
                # send F
                send_supply_position(SupplyType.MUX_BUTTON, 33, (revt_x(f_x), revt_y(f_y)))
                if DetectModeService.show_img:
                    cv2.putText(img2, 'F', (round(f_x), round(f_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            # 2. find G
            g_x = g_y = 0
            for i in range(n):
                if int(axis[i]['key_code']) == 34:
                    found, g_x, g_y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift, bf)
                    if found:
                        xc = (lst_rect[i][0] + lst_rect[i][2]) / 2
                        yc = (lst_rect[i][1] + lst_rect[i][3]) / 2
                        print(axis[i]['name'], 'found at', recvt_pos(g_x, g_y), 'center offset is',
                              (round(g_x - xc), round(g_y - yc)))
                        break

            if g_x != DetectModeService.bak_g_x or g_y != DetectModeService.bak_g_y:
                DetectModeService.bak_g_x,DetectModeService.bak_g_y = g_x, g_y
                # send G
                send_supply_position(SupplyType.MUX_BUTTON, 34, (revt_x(g_x), revt_y(g_y)))
                if DetectModeService.show_img:
                    cv2.putText(img2, 'G', (round(g_x), round(g_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            # 3. find supply
            t_s, lst_s = detect_supply(gray)
            if t_s != DetectModeService.bak_t_s or lst_s != DetectModeService.bak_lst_s:
                DetectModeService.bak_t_s, DetectModeService.bak_lst_s = t_s, lst_s
                for i in range(len(lst_s)):
                    # send new position
                    x = revt_x(lst_s[i][0])
                    y = revt_y(lst_s[i][1])
                    send_supply_position(t_s, i+1, (x, y))

                    # print result
                    if t_s == SupplyType.SUPPLY_RANDOM:
                        print('random supply', i + 1, recvt_pos(lst_s[i][0], lst_s[i][1]))
                    elif t_s == SupplyType.SUPPLY_SYSTEM:
                        print('system supply', i + 1, recvt_pos(lst_s[i][0], lst_s[i][1]))
                    elif t_s == SupplyType.SUPPLY_CUSTOM:
                        print('custom supply', i + 1, recvt_pos(lst_s[i][0], lst_s[i][1]))

                    # show result image
                    if DetectModeService.show_img:
                        w = lst_s[i][2]
                        h = lst_s[i][3]
                        a = (round(lst_s[i][0] - 0.5 * w), round(lst_s[i][1] - 0.5 * h))
                        b = (round(lst_s[i][0] + 0.5 * w), round(lst_s[i][1] + 0.5 * h))
                        cv2.rectangle(img2, a, b, (0, 255, 0), 2)
                        cv2.putText(img2, str(i + 1), (round(lst_s[i][0]), round(lst_s[i][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # confirm button position
                if t_s == SupplyType.SUPPLY_CUSTOM:
                    send_supply_position(SupplyType.MUX_BUTTON, 33, DetectModeService.self_def_confirm)
                    if DetectModeService.show_img:
                        cv2.putText(img2, 'F', cnvt_pos(DetectModeService.self_def_confirm),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 4. find select vehicle
            found, a, b = detect_select_vehicle(gray)
            if found != DetectModeService.bak_sel_vel:
                DetectModeService.bak_sel_vel = found
                if found:
                    send_supply_position(SupplyType.MUX_BUTTON, 16, DetectModeService.pos_prev)
                    send_supply_position(SupplyType.MUX_BUTTON, 18, DetectModeService.pos_next)
                    send_supply_position(SupplyType.MUX_BUTTON, 33, DetectModeService.pos_confirm)
                    if DetectModeService.show_img:
                        cv2.rectangle(2, a, b, (0, 255, 0), 2)
                        cv2.putText(img2, 'Q', cnvt_pos(DetectModeService.pos_prev), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        cv2.putText(img2, 'E', cnvt_pos(DetectModeService.pos_next), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        cv2.putText(img2, 'F', cnvt_pos(DetectModeService.pos_confirm), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255),
                                    2)

            # 5. find exchange seat
            g_x = g_y = 0
            for i in range(n):
                if int(axis[i]['key_code']) == 58:
                    found, g_x, g_y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift, bf)
                    if found:
                        xc = (lst_rect[i][0] + lst_rect[i][2]) / 2
                        yc = (lst_rect[i][1] + lst_rect[i][3]) / 2
                        print(axis[i]['name'], 'found at', recvt_pos(g_x, g_y), 'center offset is', (round(g_x - xc), round(g_y - yc)))
                        break

            if g_x != DetectModeService.bak_ex_x or g_y != DetectModeService.bak_ex_y:
                DetectModeService.bak_ex_x = g_x
                DetectModeService.bak_ex_y = g_y
                if g_x != 0 and g_y != 0:
                    # send_supply_position(SupplyType.MUX_BUTTON, 34, (round(g_x), round(g_y)))
                    if DetectModeService.show_img:
                        cv2.putText(img2, 'EX', (round(g_x), round(g_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

                    # detect veheicle type
                    s0 = time.time()
                    mod, des = detect_drive_mode(gray)
                    print('time elapse', time.time() - s0)

                    print("switch to sub mode", des)
                    send_sub_mode(mod)

            # 6. display result
            if DetectModeService.show_img:
                win_name = "result"
                cv2.namedWindow(win_name)
                cv2.moveWindow(win_name, 0, 0)
                cv2.imshow(win_name, img2)
                cv2.waitKey()
                cv2.destroyAllWindows()

            time.sleep(0.1)
            # lst_pic = ['print_vehicle.png','select_vehicle.png','select_weapon.png',
            #            'tough_on.png','tough_off.png','update_chip.png', 'moto.png',
            #            'take_drive.png', 'random_supply.png','system_supply.png',
            #            'custom_supply.png','tough_on.png','door.png']
            lst_pic = os.listdir('3')
            show_full_screen(lst_pic[idx%len(lst_pic)])
            idx += 1
            # show_full_screen('3/custom_supply.png')
            time.sleep(1)

    def start(self):
        t = threading.Thread(target=self.detect_thread, args=())
        t.daemon = True
        t.start()
        return t


def show_full_screen(name):
    image = cv2.imread('5/'+name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey(5)


def ll():
    time.sleep(1)
    s = DetectModeService()
    s.detect_thread()


if __name__ == '__main__':
    # show_full_screen('crop_none.png')
    # ll()
    # show_full_screen('select_weapon.png')
    # ll()
    # show_full_screen('update_chip.png')
    # ll()
    # show_full_screen('moto.png')
    # ll()
    # show_full_screen('select_vehicle.png')
    # ll()
    show_full_screen('print_vehicle_1.png')
    ll()
    # show_full_screen('tough_on.png')
    # ll()
    # show_full_screen('random_supply.png')
    # ll()
    # show_full_screen('custom_supply.png')
    # ll()
    # show_full_screen('system_supply.png')
    # ll()
    # show_full_screen('system_supply_1.png')
    # ll()
    # t = detect_service()
    # t.join()
    #
    # show_full_screen('take_drive.png')
    # ll()
    #
    # show_full_screen('door.png')
    # ll()
    #
    # show_full_screen('strop_on.png')
    # ll()
    #
    # show_full_screen('strop_off.png')
    # ll()
    # img1 = cv2.imread('4/' + name + '.png')
    # img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


class ModeInfo():
    def __init__(self):
        self.main_mode = MainModeType.MULTI_PLAYER
        self.sub_mode = SubModeType.NONE_SUB_MODE
        self.map_mode_on = False
        self.transparent_mode_on = False


mode_info = ModeInfo()


def main_mode_switch(mode):
    mode_info.main_mode = mode
    print("switch to main mode", mode)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, ControlEvent.MAIN_MODE)
    set_param2(data, int(mode))
    tcp_service.tcp_data_append(data)


# def sub_mode_switch():
#
#     if mode_info.sub_mode == SubModeType.NONE_SUB_MODE:
#         i = detect_sub_mode()
#         if i >= 0:
#             mode_info.sub_mode = i + SubModeType.SUB_MODE_OFFSET
#             print("switch to sub mode", mode_info.sub_mode)
#             send_sub_mode(mode_info.sub_mode)
#         else:
#             print("no sub mode detected")
#     else:
#         mode_info.sub_mode = SubModeType.NONE_SUB_MODE
#         send_sub_mode(mode_info.sub_mode)


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
