import csv
import ctypes
import os
import time
from ctypes import memmove, byref
from operator import itemgetter

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pyautogui
import win32con
import win32gui
import win32ui
from PIL import Image, ImageDraw
from PIL import ImageGrab


def window_capture(x, y, w, h, filename):
    hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    # MoniterDev = win32api.EnumDisplayMonitors(None, None)
    # w = MoniterDev[0][2][2]
    # h = MoniterDev[0][2][3]
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (x, y), win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)


# windows screenshot
def test():
    beg = time.time()
    window_capture(1000, 500, 280, 220, "haha.jpg")
    end = time.time()
    print(end - beg)


# pyautogui screenshot
def test1(name):
    beg = time.time()
    img = pyautogui.screenshot(region=[870, 350, 1280 - 870, 720 - 350])
    end = time.time()
    img.save(name)
    print(end - beg)


# screenshot find axis of picture file
def test2():
    beg = time.time()
    pos = pyautogui.locateOnScreen('coyote1.png')
    end = time.time()
    print(end - beg)
    print(pos)

    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # cv2.imshow("截屏",img)
    # cv2.waitKey(0)


# compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse, diff


# opencv screenshot compare with picture file
def test3(name):
    # load the input images
    img1 = cv2.imread(name)
    img2 = cv2.imread('coyote.png')

    beg = time.time()
    img3 = pyautogui.screenshot(region=[870, 350, 1280 - 870, 720 - 350])
    # convert the images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray' + name, img3)

    error, diff = mse(img1, img3)
    end = time.time()
    print('last', end - beg)

    print("Image matching Error between the two images:", error)


def test4(name):
    image = cv2.imread(name)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=33, maxRadius=47)
                              # param1=60, param2=40, minRadius=38, maxRadius=45)
    # If some circle is found
    if circles is not None:
        # Get the (x, y, r) as integers
        circles = np.round(circles[0, :]).astype("int")
        print(circles)
        # loop over the circles
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("circle", output)
    cv2.waitKey(0)


def test5(name):
    # use canny, as HoughCircles seems to prefer ring like circles to filled ones.
    image = cv2.imread(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    wide = cv2.Canny(blurred, 10, 200)
    mid = cv2.Canny(blurred, 30, 150)
    tight = cv2.Canny(blurred, 240, 250)
    cv2.imshow("Original", image)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Wide Edge Map", wide)
    cv2.imshow("Mid Edge Map", mid)
    cv2.imshow("Tight Edge Map", tight)
    cv2.waitKey(0)
    # smooth to reduce noise a bit more
    # cv2.Smooth(img, img, cv.CV_GAUSSIAN, 7, 7)


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


def detect_boat():
    image = cv2.imread("boat.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=33, maxRadius=37)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

    return circles


def test6():
    chopper = [[118, 86, 42]]
    moto = [[79, 190, 40], [79, 59, 41], [201, 55, 41]]
    coyote = [[84, 263, 44], [156, 69, 40], [66, 143, 44], [290, 69, 43]]
    boat = [338, 299, 35]

    name = 'boat.png'
    circles = detect_circles_of_file(name)
    # circles = detect_circles_by_capture()
    if circles is not None:
        M = [chopper, moto, coyote]
        i = get_match_item(circles, M)
        print('is', i)
    else:
        circles = detect_boate()
        if len(circles) == 1 and similar_point(circles[0], boat):
            i = 0
            print('boat')


def test7():
    # img = cv2.imread('1/door.png')
    img = cv2.imread('3/select_vehicle.png')#120
    # img = cv2.imread('1/pickup.png')
    # img = cv2.imread('1/kongtou.png')
    # img = cv2.imread('1/kongtougreen.png')
    #img = cv2.imread('1/redbox.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    idx = 0
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100:
                idx += 1
                cv2.putText(img, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                print(x, y, w, h)
            # ratio = float(w)/h
            # if ratio >= 0.9 and ratio <= 1.1:
            #    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
            #    cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # else:
            #    cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_door():
    image = cv2.imread("1/door.png")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=27, maxRadius=31)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

        # loop over the circles
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("circle", output)
    cv2.waitKey(0)
    return circles


def detect_drive():
    name = "1/drive.png"
    image = cv2.imread(name)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=42, maxRadius=46)
    idx = 0
    if circles is not None:
        # circles = np.round(circles[0, :]).astype("int")
        circles = (circles[0, :]).astype("float")

        # loop over the circles
        for (x, y, r) in circles:
            idx += 1
            cv2.circle(output, (round(x), round(y)), round(r), (0, 255, 0), 2)
            cv2.putText(output, str(idx), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            print(idx, circles[idx - 1])

    # show the output image
    cv2.imshow("circle", output)
    cv2.waitKey(0)

    return circles


def detect_tough_guy():
    image = cv2.imread("1/man.png")
    # image = cv2.imread("1/man_down.png")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=38, maxRadius=42)
    idx = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

        # loop over the circles
        for (x, y, r) in circles:
            idx += 1
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    # show the output image
    cv2.imshow("circle", output)
    cv2.waitKey(0)
    return circles


def sift_match():
    # import required libraries
    import cv2

    # read two input images as grayscale
    img1 = cv2.imread('save.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('3/print_vehicle_1.png', cv2.IMREAD_GRAYSCALE)

    #ret, img1 = cv2.threshold(img1, 100, 255, 0)
    #ret, img2 = cv2.threshold(img2, 160, 255, 0)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # detect and compute the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.match(des1, des2)

    # sort the matches based on distance
    matches = sorted(matches, key=lambda val: val.distance)

    # Draw first 50 matches.
    out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(out), plt.show()

    for i in range(10):
        print(matches[i].distance)

    print(kp2[matches[0].trainIdx].pt)


def sift_match_zone(src, name, x1, y1, x2, y2):
    # import required libraries
    show_img = False
    import cv2
    img1 = cv2.imread('4/' + name + '.png',cv2.IMREAD_UNCHANGED)
    if show_img:
        cv2.imshow('saved',img1)
        cv2.waitKey(3)
    # read two input images as grayscale
    img1 = cv2.imread('4/' + name + '.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    crop = img2[y1:y2, x1:x2]
    # print(x1, y1, x2, y2, img1.shape, crop.shape)
    # cv2.imshow("crop", crop)
    # cv2.waitKey(0)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # ret, img1 = cv2.threshold(img1, 150, 255, 0)
    # ret, crop = cv2.threshold(crop, 150, 255, 0)

    # detect and compute the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(crop, None)
    print('keypoints',len(kp1),len(kp2))
    # create BFMatcher object
    bf = cv2.BFMatcher()

    start = time.time()
    # Match descriptors.
    matches = bf.match(des1, des2)

    # sort the matches based on distance
    matches = sorted(matches, key=lambda val: val.distance)

    print('time elipse:', time.time() - start)
    # Draw first 50 matches.
    if show_img:
        out = cv2.drawMatches(img1, kp1, crop, kp2, matches[:50], None, flags=2)
        cv2.imshow(out)
        cv2.waitKey()

    for i in range(10):
        print(matches[i].distance, x1 + kp2[matches[i].trainIdx].pt[0], y1 + kp2[matches[i].trainIdx].pt[1])

    if matches[0].distance < 10:
        return True, x1 + kp2[matches[0].trainIdx].pt[0], y1 + kp2[matches[0].trainIdx].pt[1]
    else:
        return False, 0, 0


def crop_circle():
    # Open the input image as numpy array, convert to RGB
    img = Image.open("1/drive.png")
    img = img.crop((100, 100, 200, 200)).convert("RGB")
    img.show()
    npImage = np.array(img)
    h, w = img.size

    # Create same size alpha layer with circle
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    # draw.pieslice([0, 0, h, w], 0, 360, fill=255)
    draw.pieslice(((0, 0), (h, w)), 0, 360, fill=255)

    # Convert alpha Image to numpy array
    npAlpha = np.array(alpha)

    # Add alpha layer to RGB
    npImage = np.dstack((npImage, npAlpha))

    # Save with alpha
    Image.fromarray(npImage).save('1/result.png')


def save_circle_panel(src, x, y, r, dst):
    img = Image.open(src)
    img = img.crop((x - r - 2, y - r - 2, x + r + 2, y + r + 2)).convert("RGB")
    # img.show()
    npImage = np.array(img)
    h, w = img.size

    # Create same size alpha layer with circle
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((0, 0), (h, w)), 0, 360, fill=255)

    # Convert alpha Image to numpy array
    npAlpha = np.array(alpha)

    # Add alpha layer to RGB
    npImage = np.dstack((npImage, npAlpha))

    # Save with alpha
    Image.fromarray(npImage).save(dst)

    image = Image.open(dst)
    image.convert("RGBA")
    canvas = Image.new('RGBA', image.size, (255, 255, 255, 255))  # Empty canvas colour (r,g,b,a)
    canvas.paste(image, mask=image)  # Paste the image onto the canvas, using it's alpha channel as mask
    canvas.save(dst, format="PNG")


def save_panel_axis(key_code, id, name, left, top, right, bottom):
    dct = {}
    dct['key_code'] = key_code
    dct['id'] = id
    dct['name'] = name
    dct['left'] = left
    dct['top'] = top
    dct['right'] = right
    dct['bottom'] = bottom
    lst = read_panel_axis()
    for i in range(len(lst)):
        if lst[i]['name'] == name:
            del lst[i]
            break
    lst.append(dct)
    lst = sorted(lst, key=lambda d: int(d['id']))
    lst = sorted(lst, key=lambda d: int(d['key_code']))

    with open('panels.csv', 'w', newline='\n') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, fieldnames=dct.keys())
        w.writeheader()

        for i in lst:
            w.writerow(i)
        f.close()


def read_panel_axis():
    with open('panels.csv', 'r', newline='\n') as f:
        reader = csv.DictReader(f)
        # print(reader.line_num)
        lst = [*reader]
        # print(lst)
        # for row in reader:
        #     print(row.get('name'))
        f.close()
        lst = sorted(lst, key=lambda d: int(d['id']))
        lst = sorted(lst, key=lambda d: int(d['key_code']))
        return lst


def detect_circle_panel(file_name, min_radius, max_radius, tag):
    image = cv2.imread(file_name)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
                               param1=60, param2=40, minRadius=min_radius, maxRadius=max_radius)
    idx = 0
    if circles is not None:
        # circles = np.round(circles[0, :]).astype("int")
        circles = (circles[0, :]).astype("float")

        for (x, y, r) in circles:
            idx += 1
            cv2.circle(output, (round(x), round(y)), round(r), (0, 255, 0), 2)
            cv2.putText(output, str(idx), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            print(idx, circles[idx - 1])

        # show the output image
        cv2.imshow("select "+tag, output)
        ret = cv2.waitKey(0)
        ret = int(ret - 0x30)
        cv2.destroyAllWindows()

        if ret > 0:
            return True, circles[ret - 1][0], circles[ret - 1][1], circles[ret - 1][2]
        else:
            return False, 0 ,0 ,0
    else:
        return False, 0, 0, 0


def select_and_sift_match(key_code, id, name, min_radius, max_radius):
    file = '3/'+name+'.png'
    b, x, y, r = detect_circle_panel(file, min_radius, max_radius, name)
    if not b:
        return
    xc, yc = x, y
    save_circle_panel(file, x, y, r, '4/' + name + '.png')
    save_panel_axis(key_code, id, name, x - r, y - r, x + r, y + r)

    axis = read_panel_axis()

    for d in axis:
        if d['name'] == name:
            found, x, y = sift_match_zone(file, name, round(float(d['left'])) - 2, round(float(d['top'])) - 2,
                                          round(float(d['right'])) + 2, round(float(d['bottom'])) + 2)
            if found:
                print(name, 'found at', (x, y), 'center offset is', (x - xc, y - yc))

def detect_rectangle():
    img = cv2.imread('d:/Untitled.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 50, 255, 0)
    cv2.imshow('image', thresh)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    lst = []
    self_def = False
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >20:
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                print(w,h)

    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detect_supply_box():
    # img = cv2.imread('1/selectcar.png')#120
    img = cv2.imread('3/custom_supply.png')
    #img = cv2.imread('3/system_supply.png')
    #img = cv2.imread('3/system_supply_1.png')
    left = 368
    right = 877
    top = 286
    bottom = 571

    target_width = 216
    target_high = 61

    self_def_width = 248
    self_def_high = 176

    crop = img[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    # cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    lst = []
    self_def = False
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                lst.append((x, y, w, h))
            elif abs(self_def_width - w) <= 3 and abs(self_def_high - h) <= 3:
                crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                self_def = True
                print(x, y, w, h)

    if self_def:
        cv2.putText(img, str(1), (left + round(0.5 * self_def_width), top + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(img, str(2), (left + round(1.5 * self_def_width), top + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(img, str(3), (left + round(0.5 * self_def_width), bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(img, str(4), (left + round(1.5 * self_def_width), bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        # confirm (740,636)
    else:
        lst = sorted(lst, key=itemgetter(0))
        lst = sorted(lst, key=itemgetter(1))
        for i in range(len(lst)):
            cv2.putText(img, str(i + 1),
                        (left + lst[i][0] + round(0.5 * target_width), top + lst[i][1] + round(0.5 * target_high)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(lst)

    cv2.imshow("Shapes", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_random_supply():
    img = cv2.imread('3/random_supply.png')

    left = 681
    right = 933
    top = 283
    bottom = 571
    target_width = 216
    target_high = 60

    crop = img[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, 0)
    # cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    lst = []
    self_def = False
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(target_width - w) <= 3 and abs(target_high - h) <= 3:
                crop = cv2.drawContours(crop, [cnt], -1, (0, 255, 0), 2)
                lst.append((x, y, w, h))

    lst = sorted(lst, key=itemgetter(1))
    for i in range(len(lst)):
        cv2.putText(img, str(i + 1),
                    (left + lst[i][0] + round(0.5 * target_width), top + lst[i][1] + round(0.5 * target_high)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(lst)

    cv2.imshow("Shapes", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_dict():
    dt = {}
    dt['txt'] = 1
    dt['rect'] = (100, 80, 200, 50)
    lst = [dt, dt]

    print(lst)

def update_screen():

    t0 = time.time()
    while True:
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        crop = screenshot[385:476, 891:983]
        cv2.imshow("Computer Vision", crop)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        ex_time = time.time() - t0
        print("FPS: " + str(1 / (ex_time)))
        t0 = time.time()

def folder_resize():
    lst = os.listdir('1')
    for name in lst:
        print(name)
        resize_image(name)
def resize_image(name):
    image = cv2.imread('1/'+name)
    resized = cv2.resize(image, (1280,720), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('3/'+name,resized)


def save_screenshot(name):
    #show_full_screen('1/'+name)
    img2 = pyautogui.screenshot(region=(0, 0, 1280, 720))
    img2.save('3/'+name)


def byte_copy():
    data = [ctypes.c_byte()]*4
    data[0] = 1
    data[1] = 2
    data[2] = 3
    data[3] = 4

    t = ctypes.c_int32(0)
    a = ctypes.c_byte(1)
    b = ctypes.c_byte(2)
    memmove(byref(t, 0), byref(a), 1)
    memmove(byref(t, 2), byref(b), 1)

    print(t)
    a = ctypes.c_int16(0x12)
    b = ctypes.c_int16(0x34)
    memmove(byref(t, 0), byref(a), 2)
    memmove(byref(t, 2), byref(b), 2)
    print(t)

def mod_png():
   img = cv2.imread('save.png',cv2.IMREAD_GRAYSCALE)
   ret, thresh = cv2.threshold(img, 120, 255, 0)
   cv2.imshow('img', thresh)
   cv2.waitKey()
   cv2.destroyAllWindows()
def cut_image():
    img = cv2.imread('3/none.png')
    crop = img[0:720, 0:1280]
    cv2.imwrite('3/crop_none.png', crop)
def find_homo():
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    MIN_MATCH_COUNT = 10
    img1 = cv.imread('4/print_vehicle.png', cv.IMREAD_GRAYSCALE)  # queryImage
    #img1 = cv.imread('4/print_vehicle.png', cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread('3/print_vehicle_1.png', cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print('good:', len(good), len(kp1), len(kp2), len(matches), len(good)/len(matches))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

def paste_image():
    files = os.listdir('3')
    for f in files:
        img = cv2.imread('3/'+f)
        resized = cv2.resize(img, (1213, 682), interpolation=cv2.INTER_LINEAR)
        img2 = numpy.zeros((720,1280,3), numpy.uint8)
        img2[38:720,32:1245] = resized
        cv2.imwrite('5/'+f, img2)

def create_pannel_csv():
    select_and_sift_match(33, 0, 'take_drive',  40, 44)
    select_and_sift_match(33, 1, 'door',  25, 29)
    select_and_sift_match(33, 2, 'strop_on',  35, 39)
    select_and_sift_match(33, 3, 'strop_off',  35, 39)
    select_and_sift_match(33, 4, 'tough_on',  35, 39)
    select_and_sift_match(33, 5, 'tough_off',  35, 39)
    select_and_sift_match(33, 6, 'print_vehicle',  20, 35)
    select_and_sift_match(33, 7, 'update_chip',  20, 35)
    select_and_sift_match(33, 8, 'select_weapon',  36, 40)
    select_and_sift_match(34, 0, 'drive_by',  40, 44)
    select_and_sift_match(58, 0, 'ex_seat',  32, 36)

find_homo()

#detect_rectangle()
#cut_image()
#test4('3/boat.png')
#test4('moto.png')
#sift_match()
#find_homo()

#byte_copy()
#folder_resize()
#resize_image('redbox.png')
#update_screen()
# detect_supply_box()
# detect_random_supply()
#test_dict()
# sift_match_zone('1/redbox.png', 'suply', 392, 302, 430, 600)
# sift_match_zone('1/redbox.png', 'suply', 392, 440, 430, 600)

if __name__ == '__':
    test7()
    detect_door()
    detect_drive()
    detect_tough_guy()
