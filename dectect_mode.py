import csv
import ctypes
import threading
import time
from collections import OrderedDict
from operator import itemgetter

import cv2
import numpy as np
import pyautogui

import switch_mode
import tcp_service
from defs import TcpData, EventType, set_param1_int32, set_param2_int32, ControlEvent, \
    SubModeType, SupplyType, MainModeType, MotionType
from window_info import get_window_info, get_terminal_size


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
        # print('item', i, 'match', m)

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
    dc = {SubModeType.NONE_SUB_MODE: 'none', SubModeType.DRIVE_MOTO: 'moto',
          SubModeType.DRIVE_CHOPPER: 'chopper', SubModeType.DRIVE_COYOTE: 'coyote'}

    crop = img[round(cnvt_x(350)):DetectModeService.wd_height, round(cnvt_x(870)):DetectModeService.wd_width]
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=int(cnvt_x(38)), maxRadius=int(cnvt_x(45)))
    i = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        i = get_match_item(circles, M)
        # print('is', i)
    else:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                                   param1=60, param2=40, minRadius=33, maxRadius=37)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(circles)
            if len(circles) == 1:
                if similar_point(circles[0], boat[0]):
                    i = 0

    i += SubModeType.SUB_MODE_OFFSET  # print('boat')
    return i, dc[i]


def send_sub_mode(mod):
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1_int32(data, ControlEvent.SUB_MODE)
    set_param2_int32(data, mod)
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

    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda val: val.distance)

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
        # out = cv2.drawMatches(img1, kp1, crop, kp2, matches[:5], None, flags=2)
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
        axis = sorted(axis, key=lambda d: int(d['id']))
        axis = sorted(axis, key=lambda d: int(d['key_code']))

        keys = []
        for a in axis:
            keys.append(a['key_code'])
        keys = list(dict.fromkeys(keys))

        lst = OrderedDict([])
        for k in keys:
            lst[k] = []
            for a in axis:
                if a['key_code'] == k:
                    b = a.copy()
                    b.pop('key_code')
                    lst[k].append(b)
        return lst


def send_supply_position(v_key_code, v_pos=(0, 0), v_motion=MotionType.MOTION_SYNC, v_vector=(0, 0)):
    print('send key', v_key_code, 'at', v_pos)
    data = TcpData()
    data.type = EventType.TYPE_SET_KEY_MOTION

    v = ctypes.c_uint8(v_key_code)
    ctypes.memmove(ctypes.byref(data, TcpData.param1.offset), ctypes.byref(v), 1)
    v = ctypes.c_uint8(v_motion)
    ctypes.memmove(ctypes.byref(data, TcpData.param1.offset + 1), ctypes.byref(v), 1)
    v = ctypes.c_int8(v_vector[0])
    ctypes.memmove(ctypes.byref(data, TcpData.param1.offset + 2), ctypes.byref(v), 1)
    v = ctypes.c_int8(v_vector[1])
    ctypes.memmove(ctypes.byref(data, TcpData.param1.offset + 3), ctypes.byref(v), 1)

    v = ctypes.c_uint16(v_pos[0])
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset), ctypes.byref(v), 2)
    v = ctypes.c_uint16(v_pos[1])
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset + 2), ctypes.byref(v), 2)
    tcp_service.tcp_data_append(data)


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
    supply_type = SupplyType.SUPPLY_NONE
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


def detect_main_mode(img):
    main_mode = -1
    found, _, _ = detect_rectangle_contour(img, 0, 0, 175, 40, 155, 32)
    if found:
        main_mode = MainModeType.BATTLE_GROUND
    return main_mode


def detect_supply(img):
    # random supply religion and size
    left, top, right, bottom = [int(i) for i in cnvt_rect(681, 283, 933, 571)]
    target_width, target_high = cnvt_size(216, 60)

    crop = img[top:bottom, left:right]
    ret, thresh = cv2.threshold(crop, 120, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    supply_type = SupplyType.SUPPLY_NONE
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                lst.append((left + x + 0.5 * w, top + y + 0.5 * h, w, h))
                supply_type = SupplyType.SUPPLY_RANDOM

    if supply_type == SupplyType.SUPPLY_NONE:
        supply_type, lst = detect_supply_box(img)
    else:
        lst = sorted(lst, key=itemgetter(1))

    # zeros empty position
    for i in range(6 - len(lst)):
        lst.append((0, 0, 0, 0))

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


def detect_rectangle_contour(origin, x1, y1, x2, y2, w, h):
    left, top, right, bottom = [int(i) for i in cnvt_rect(x1, y1, x2, y2)]
    target_width, target_high = cnvt_size(w, h)

    img = cv2.cvtColor(np.array(origin), cv2.COLOR_RGB2BGR)
    img = img[top:bottom, left:right]

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # img_rsz = cv2.resize(closing, (800, 400))
    # cv2.imshow('closing', img_rsz)
    # cv2.waitKey(100)

    contours, hierarchy = cv2.findContours(closing, 1, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    a, b = (0, 0), (0, 0)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                found = True
                a, b = ((left + x, top + y), (left + x + w, top + y + h))
                # print('rectangle found at', (left + x, top + y))
                break

    return found, a, b


def detect_rectangle_grabcut(origin, x1, y1, x2, y2, w, h):
    left, top, right, bottom = [int(i) for i in cnvt_rect(x1, y1, x2, y2)]
    target_width, target_high = cnvt_size(w, h)

    img = cv2.cvtColor(np.array(origin), cv2.COLOR_RGB2BGR)
    img = img[top:bottom, left:right]
    mask = np.zeros(img.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1] - 1, img.shape[0] - 1)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    img_rsz = cv2.resize(img, (800, 400))
    cv2.imshow('grab_cut', img_rsz)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rsz = cv2.resize(gray, (800, 400))
    cv2.imshow('gray', img_rsz)

    kernel = np.ones((6, 6), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    img_rsz = cv2.resize(closing, (800, 400))
    cv2.imshow('closing', img_rsz)

    thresh = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    img_rsz = cv2.resize(thresh, (800, 400))
    cv2.imshow('thresh', img_rsz)

    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    a, b = (0, 0), (0, 0)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                found = True
                a, b = ((left + x, top + y), (left + x + w, top + y + h))
                # print('rectangle found at', (left + x, top + y))
                break

    cv2.waitKey()

    return found, a, b


def cnvt_x(x):
    return x * DetectModeService.wd_width / DetectModeService.mc_width


def cnvt_y(y):
    return y * DetectModeService.wd_height / DetectModeService.mc_height


def cnvt_pos(p):
    return round(cnvt_x(p[0])), round(cnvt_y(p[1]))


def cnvt_region(x, y, w, h):
    return DetectModeService.wd_left + cnvt_x(x), DetectModeService.wd_top + cnvt_y(y), cnvt_x(w), cnvt_y(h)


def cnvt_rect(left, top, right, bottom):
    return cnvt_x(left), cnvt_y(top), cnvt_x(right), cnvt_y(bottom)


def cnvt_size(width, high):
    return cnvt_x(width), cnvt_y(high)


def cnvt_circles(circles):
    return [[cnvt_x(c[0]), cnvt_y(c[1]), cnvt_x(c[2])] for c in circles]


def revt_x(x):
    return round(x * DetectModeService.mc_width / DetectModeService.wd_width)


def revt_y(y):
    return round(y * DetectModeService.mc_height / DetectModeService.wd_height)


def recvt_pos(x, y):
    return round(revt_x(x)), round(revt_y(y))


class DetectModeService:
    show_img = False
    custom_confirm = (693, 596)  # red supply enter
    supply_box_close = (851, 217)
    random_supply_close = (913, 217)
    pos_prev = (500, 500)  # prev vehicle
    pos_next = (775, 500)  # next vehicle
    pos_confirm = (636, 542)  # vehicle confirm
    bak_f_x, bak_f_y = (0, 0)
    bak_g_x, bak_g_y = (0, 0)
    bak_ex_x, bak_ex_y = (0, 0)

    bak_t_s = SupplyType.SUPPLY_NONE
    bak_lst_s = [(0, 0, 0, 0) for i in range(6)]
    bak_sel_vel = False

    bak_armor_off = False

    f_found = [False] * 4
    bak_f_found = [False] * 4

    bak_main_mode = -1
    # 32 38 1213 682
    wd_left, wd_top, wd_width, wd_height = 0, 0, 1280, 720
    mc_width, mc_height = 1280, 720

    axis = read_panel_axis()
    b_found = dict()
    pos_x = dict()
    pos_y = dict()
    for a in axis:
        b_found[a] = False
        pos_x[a] = 0
        pos_y[a] = 0

    def __init__(self):
        # 32 38 1213 682
        DetectModeService.wd_left, DetectModeService.wd_top, \
            DetectModeService.wd_width, DetectModeService.wd_height = get_window_info()
        DetectModeService.mc_width, DetectModeService.mc_height = get_terminal_size()

    def detect_thread(self):

        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

        lsg_img = {}
        lst_rect = {}
        lst_kp = {}
        lst_des = {}

        for a in DetectModeService.axis:
            for d in DetectModeService.axis[a]:
                name = d['name']
                img1 = cv2.imread('4/' + name + '.png', cv2.IMREAD_GRAYSCALE)
                lsg_img[name] = img1
                left, top, right, bottom = float(d['left']), float(d['top']), float(d['right']), float(d['bottom'])
                lst_rect[name] = cnvt_rect(left, top, right, bottom)
                kp1, des1 = sift.detectAndCompute(img1, None)
                lst_kp[name] = kp1
                lst_des[name] = des1
                # print(name, 'keypoint num', len(kp1))

        n_mode = 0
        while True:
            r = [int(i) for i in cnvt_region(0, 0, DetectModeService.mc_width, DetectModeService.mc_height)]
            img = pyautogui.screenshot(region=r)
            img2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # check main made
            v_mode = detect_main_mode(img)
            if v_mode != -1:
                if v_mode != DetectModeService.bak_main_mode:
                    DetectModeService.bak_main_mode = v_mode
                    n_mode = 0
                else:
                    n_mode += 1
                    if n_mode == 10:
                        n_mode = 0
                        if switch_mode.ModeInfo.main_mode != v_mode:
                            switch_mode.ModeInfo.main_mode = v_mode
                            switch_mode.main_mode_switch(switch_mode.ModeInfo.main_mode)

            # battleground mode
            if switch_mode.ModeInfo.main_mode == MainModeType.BATTLE_GROUND:
                # 1. find alternate  panels

                for a in DetectModeService.axis:
                    f_x, f_y = 0, 0
                    found = False
                    f_name = ''

                    for d in DetectModeService.axis[a]:
                        f_name = d['name']
                        method = int(d['detect_method'])
                        if method == 0:
                            found, f_x, f_y = sift_match_zone(lsg_img[f_name], gray, lst_rect[f_name], lst_kp[f_name],
                                                              lst_des[f_name], sift,
                                                              bf)
                        elif method == 1:
                            x1, y1, x2, y2 = lst_rect[f_name]
                            edge = 6
                            found, m, n = detect_rectangle_grabcut(img, x1, y1, x2, y2, x2 - x1,
                                                                   y2 - y1)
                            f_x = round((m[0] + n[0]) / 2)
                            f_y = round((m[1] + n[1]) / 2)
                        elif method == 2:
                            x1, y1, x2, y2 = lst_rect[f_name]
                            found, m, n = detect_rectangle_contour(img, x1, y1, x2, y2, x2 - x1,
                                                                   y2 - y1)
                            f_x = round((m[0] + n[0]) / 2)
                            f_y = round((m[1] + n[1]) / 2)
                        if found:
                            break

                    DetectModeService.f_found[0] = found

                    if abs(f_x - DetectModeService.pos_x[a]) > 10 or abs(f_y - DetectModeService.pos_y[a]) > 10:
                        DetectModeService.pos_x[a] = f_x
                        DetectModeService.pos_y[a] = f_y

                        # print
                        if found:
                            xc = (lst_rect[f_name][0] + lst_rect[f_name][2]) / 2
                            yc = (lst_rect[f_name][1] + lst_rect[f_name][3]) / 2
                            print(f_name, 'found at', recvt_pos(f_x, f_y), 'center offset is',
                                  (round(f_x - xc), round(f_y - yc)))

                            # send panel position
                            send_supply_position(int(a), (revt_x(f_x), revt_y(f_y)))
                            if DetectModeService.show_img:
                                cv2.putText(img2, a, (round(f_x), round(f_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)

                        # drive mode switch
                        if f_name == 'ex_seat':
                            if found:
                                # detect vehicle type
                                # s0 = time.time()
                                mod, des = detect_drive_mode(gray)
                                # print('time elapse', time.time() - s0)

                                if SubModeType.NONE_SUB_MODE == mod:
                                    # type not recognized
                                    DetectModeService.bak_ex_x = 0
                                    DetectModeService.bak_ex_y = 0
                                else:
                                    print("drive on", des)
                                    send_sub_mode(mod)
                            else:
                                print("drive off")
                                send_sub_mode(SubModeType.NONE_SUB_MODE)
                #
                # # 2. find G panels
                # j, g_x, g_y = 0, 0, 0
                # found = False
                # for i in range(n):
                #     if int(axis[i]['key_code']) == 34:
                #         found, g_x, g_y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift,
                #                                           bf)
                #         if found:
                #             j = i
                #             break
                #
                # if abs(g_x - DetectModeService.bak_g_x) > 10 or abs(g_y - DetectModeService.bak_g_y) > 10:
                #     DetectModeService.bak_g_x, DetectModeService.bak_g_y = g_x, g_y
                #     # print
                #     if found:
                #         xc = (lst_rect[j][0] + lst_rect[j][2]) / 2
                #         yc = (lst_rect[j][1] + lst_rect[j][3]) / 2
                #         print(axis[j]['name'], 'found at', recvt_pos(g_x, g_y), 'center offset is',
                #               (round(g_x - xc), round(g_y - yc)))
                #     # send G
                #     send_supply_position(34, (revt_x(g_x), revt_y(g_y)))
                #     if DetectModeService.show_img:
                #         cv2.putText(img2, 'G', (round(g_x), round(g_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                     (0, 0, 255), 2)

                # 3. find supply
                t_s, lst_s = detect_supply(gray)

                if t_s == SupplyType.SUPPLY_CUSTOM:
                    DetectModeService.f_found[1] = True
                else:
                    DetectModeService.f_found[1] = False

                if t_s != DetectModeService.bak_t_s or lst_s != DetectModeService.bak_lst_s:

                    for i in range(len(lst_s)):
                        if lst_s[i] != DetectModeService.bak_lst_s[i]:
                            # send new position
                            x = revt_x(lst_s[i][0])
                            y = revt_y(lst_s[i][1])

                            # keys list [H J K L ; ']
                            send_supply_position(i + 35, (x, y))

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

                    if t_s != DetectModeService.bak_t_s:
                        if t_s == SupplyType.SUPPLY_RANDOM:
                            # random ESC
                            send_supply_position(1, DetectModeService.random_supply_close)
                            # slide up
                            send_supply_position(180, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(0 - lst_s[0][3])))
                            # slide down
                            send_supply_position(181, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(lst_s[0][3])))
                            if DetectModeService.show_img:
                                cv2.putText(img2, 'ESC', cnvt_pos(DetectModeService.random_supply_close),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif t_s == SupplyType.SUPPLY_SYSTEM:
                            # system ESC
                            send_supply_position(1, DetectModeService.supply_box_close)
                            # slide up
                            send_supply_position(180, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(0 - lst_s[0][3])))
                            # slide down
                            send_supply_position(181, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(lst_s[0][3])))
                            if DetectModeService.show_img:
                                cv2.putText(img2, 'ESC', cnvt_pos(DetectModeService.supply_box_close),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif t_s == SupplyType.SUPPLY_CUSTOM:
                            # custom ESC and F
                            send_supply_position(1, DetectModeService.supply_box_close)
                            send_supply_position(33, DetectModeService.custom_confirm)
                            # slide up
                            send_supply_position(180, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(0 - cnvt_y(176))))
                            # slide up
                            send_supply_position(181, (round(lst_s[0][0]), round(lst_s[0][1])),
                                                 MotionType.MOTION_DRAG, (0, round(cnvt_y(176))))
                            if DetectModeService.show_img:
                                cv2.putText(img2, 'ESC', cnvt_pos(DetectModeService.supply_box_close),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(img2, 'F', cnvt_pos(DetectModeService.custom_confirm),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # none ESC. Later F
                        elif t_s == SupplyType.SUPPLY_NONE:
                            send_supply_position(1)
                            send_supply_position(180)
                            send_supply_position(181)

                    DetectModeService.bak_t_s, DetectModeService.bak_lst_s = t_s, lst_s
                #
                # # 4. find select vehicle
                # # found, a, b = detect_select_vehicle(gray)
                # found, a, b = detect_rectangle_grabcut(img, 536, 524, 736, 560, 193, 31)
                #
                # DetectModeService.f_found[2] = found
                #
                # if found != DetectModeService.bak_sel_vel:
                #     DetectModeService.bak_sel_vel = found
                #     if found:
                #         send_supply_position(16, DetectModeService.pos_prev)
                #         send_supply_position(18, DetectModeService.pos_next)
                #         send_supply_position(33, DetectModeService.pos_confirm)
                #         if DetectModeService.show_img:
                #             cv2.rectangle(2, a, b, (0, 255, 0), 2)
                #             cv2.putText(img2, 'Q', cnvt_pos(DetectModeService.pos_prev), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                         (0, 0, 255), 2)
                #             cv2.putText(img2, 'E', cnvt_pos(DetectModeService.pos_next), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                         (0, 0, 255), 2)
                #             cv2.putText(img2, 'F', cnvt_pos(DetectModeService.pos_confirm), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                         (0, 0, 255),
                #                         2)
                #     else:
                #         # disable alter keys
                #         send_supply_position(16)
                #         send_supply_position(18)
                #
                # # 5. find armor off
                # found, a, b = detect_rectangle_grabcut(img, 580, 460, 710, 520, 96, 35)
                # DetectModeService.f_found[3] = found
                #
                # if found != DetectModeService.bak_armor_off:
                #     DetectModeService.bak_armor_off = found
                #     if found:
                #         x = round((a[0] + b[0]) / 2)
                #         y = round((a[1] + b[1]) / 2)
                #
                #         print('armor off FOUND')
                #         send_supply_position(33, (x, y))
                #         if DetectModeService.show_img:
                #             cv2.rectangle(2, a, b, (0, 255, 0), 2)
                #             cv2.putText(img2, 'F', cnvt_pos((x, y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                         (0, 0, 255), 2)
                #     else:
                #         print('armor off NOT found')
                #
                # #  every F panel disappeared
                # if DetectModeService.f_found != DetectModeService.bak_f_found:
                #     DetectModeService.bak_f_found = DetectModeService.f_found
                #     if DetectModeService.f_found == [False] * 4:
                #         send_supply_position(33)
                #
                # # 6. find exchange seat
                # j, g_x, g_y = 0, 0, 0
                # found = False
                # for i in range(n):
                #     if int(axis[i]['key_code']) == 58:
                #         found, g_x, g_y = sift_match_zone(lsg_img[i], gray, lst_rect[i], lst_kp[i], lst_des[i], sift,
                #                                           bf)
                #         if found:
                #             j = i
                #             break
                #
                # if abs(g_x - DetectModeService.bak_ex_x) > 10 or abs(g_y - DetectModeService.bak_ex_y) > 10:
                #     DetectModeService.bak_ex_x = g_x
                #     DetectModeService.bak_ex_y = g_y
                #     if found:
                #         # print
                #         xc = (lst_rect[j][0] + lst_rect[j][2]) / 2
                #         yc = (lst_rect[j][1] + lst_rect[j][3]) / 2
                #         print(axis[j]['name'], 'found at', recvt_pos(g_x, g_y), 'center offset is',
                #               (round(g_x - xc), round(g_y - yc)))
                #
                #         if DetectModeService.show_img:
                #             cv2.putText(img2, 'EX', (round(g_x), round(g_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                         (0, 0, 255), 2)
                #
                #         # detect vehicle type
                #         # s0 = time.time()
                #         mod, des = detect_drive_mode(gray)
                #         # print('time elapse', time.time() - s0)
                #
                #         if SubModeType.NONE_SUB_MODE == mod:
                #             # type not recognized
                #             DetectModeService.bak_ex_x = 0
                #             DetectModeService.bak_ex_y = 0
                #         else:
                #             print("drive on", des)
                #             send_sub_mode(mod)
                #     else:
                #         print("drive off")
                #         send_sub_mode(SubModeType.NONE_SUB_MODE)

                # 6. display result
                if DetectModeService.show_img:
                    win_name = "result"
                    cv2.namedWindow(win_name)
                    cv2.moveWindow(win_name, 0, 0)
                    cv2.imshow(win_name, img2)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

            time.sleep(0.1)

    def start(self):
        th = threading.Thread(target=self.detect_thread, args=())
        th.daemon = True
        th.start()
        return th


def show_full_screen(name):
    image = cv2.imread('5/' + name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey(5)


if __name__ == '__main__':
    # show_full_screen('print_vehicle_1.png')
    # time.sleep(1)

    t = DetectModeService().start()
    t.join()
    # read_panel_axis()
