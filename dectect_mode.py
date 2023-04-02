import csv
import threading
import time

import cv2
import numpy as np
import pyautogui


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


def detect_sub_mode():
    moto = [[79, 190, 40], [79, 59, 41], [201, 55, 41]]
    chopper = [[118, 86, 42]]
    coyote = [[84, 263, 44], [156, 69, 40], [66, 143, 44], [290, 69, 43]]
    M = [moto, chopper, coyote]
    boat = [338, 299, 35]

    circles = detect_circles_by_capture()
    i = -1
    if circles is not None:
        i = get_match_item(circles, M)
        print('is', i)
    else:
        circles = detect_boate()
        if len(circles) == 1 and similar_point(circles[0], boat):
            i = 0
            print('boat')
    return i


def sift_match_zone(img1, img2, rect, kp1, des1, sift, bf):
    x1, y1, x2, y2 = rect
    x1 = round(float(x1)) - 2
    y1 = round(float(y1)) - 2
    x2 = round(float(x2)) + 2
    y2 = round(float(y2)) + 2
    crop = img2[y1:y2, x1:x2]

    # cv2.imshow("crop",crop)
    # cv2.waitKey(0)
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
    # out = cv2.drawMatches(img1, kp1, crop, kp2, matches[:5], None, flags=2)
    # plt.imshow(out), plt.show()

    # for i in range(5):
    #     print(matches[i].distance, x1 + kp2[matches[i].trainIdx].pt[0], y1 + kp2[matches[i].trainIdx].pt[1])

    if matches[0].distance < 10:
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

    while True:

        img2 = pyautogui.screenshot(region=(0, 0, 1280, 720))
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        n = len(axis)
        for i in range(n):
            found, x, y = sift_match_zone(lsg_img[i], img2, lst_rect[i], lst_kp[i], lst_des[i], sift, bf)
            if found:
                xc = (lst_rect[i][0] + lst_rect[i][2]) / 2
                yc = (lst_rect[i][1] + lst_rect[i][3]) / 2
                print(axis[i]['name'], 'found at', (x, y), 'center offset is', (x - xc, y - yc))
                # send F point
            else:
                print(axis[i]['name'], 'not found')

        time.sleep(0.1)


def detect_service():
    t = threading.Thread(target=detect_thread, args=())
    t.daemon = True
    t.start()


def show_full_screen(name):
    image = cv2.imread(name)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', image)
    cv2.waitKey(5)

def ll():
    time.sleep(3)
    #detect_thread()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_service()
    show_full_screen('3/take_drive.png')
    ll()

    show_full_screen('3/door.png')
    ll()

    show_full_screen('3/strop_on.png')
    ll()

    show_full_screen('3/strop_off.png')
    ll()
    # img1 = cv2.imread('4/' + name + '.png')
    # img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
