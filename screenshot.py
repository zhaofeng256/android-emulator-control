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


def detect_circle(name):
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

        # loop over the circles
        idx = 0
        for (x, y, r) in circles:
            print(idx + 1, circles[idx])
            idx += 1
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, str(idx), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
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
        circles = detect_boat()
        if len(circles) == 1 and similar_point(circles[0], boat):
            i = 0
            print('boat')


def test7():
    # img = cv2.imread('1/door.png')
    img = cv2.imread('3/select_vehicle.png')  # 120
    # img = cv2.imread('1/pickup.png')
    # img = cv2.imread('1/kongtou.png')
    # img = cv2.imread('1/kongtougreen.png')
    # img = cv2.imread('1/redbox.png')
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


def sift_match(img1, img2):
    # import required libraries
    import cv2

    # read two input images as grayscale
    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    # ret, img1 = cv2.threshold(img1, 100, 255, 0)
    # ret, img2 = cv2.threshold(img2, 160, 255, 0)

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

    for i in range(len(matches)):
        print(matches[i].distance)

    print(kp2[matches[0].trainIdx].pt)


def sift_match_zone(src, name, x1, y1, x2, y2):
    # import required libraries
    show_img = True
    import cv2
    img1 = cv2.imread('4/' + name + '.png', cv2.IMREAD_UNCHANGED)
    # if show_img:
    #     cv2.imshow('saved', img1)
    #     cv2.waitKey(3)
    # read two input images as grayscale
    img2 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    crop = img2[y1:y2, x1:x2]
    # print(x1, y1, x2, y2, img1.shape, crop.shape)
    # cv2.imshow("crop", crop)
    # cv2.waitKey(0)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # ret, img1 = cv2.threshold(img1, 150, 255, 0)
    # ret, crop = cv2.threshold(crop, 150, 255, 0)

    # detect and compute the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(crop, None)
    print('key points', len(kp1), len(kp2))
    # create BFMatcher object
    bf = cv2.BFMatcher()

    start = time.time()
    # Match descriptors.
    matches = bf.match(des1, des2)

    # sort the matches based on distance
    matches = sorted(matches, key=lambda val: val.distance)
    print('time elapse:', time.time() - start)

    if len(matches) == 0:
        return False, 0, 0
    # Draw first 50 matches.
    if show_img:
        out = cv2.drawMatches(img1, kp1, crop, kp2, matches, None, flags=2)
        img_rsz = cv2.resize(out, (800, 400))
        cv2.imshow(name, img_rsz)
        cv2.waitKey()

        for i in range(len(matches)):
            if matches[i].distance < 10:
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


def save_panel_axis(key_code, v_id, v_name, left, top, right, bottom, detect_method):
    dct = {}
    dct['key_code'] = key_code
    dct['id'] = v_id
    dct['name'] = v_name
    dct['left'] = left
    dct['top'] = top
    dct['right'] = right
    dct['bottom'] = bottom
    dct['detect_method'] = detect_method
    lst = read_panel_axis()
    for i in range(len(lst)):
        if lst[i]['name'] == v_name:
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
        cv2.imshow("select " + tag, output)
        ret = cv2.waitKey(0)
        ret = int(ret - 0x30)
        cv2.destroyAllWindows()

        if ret > 0:
            return True, circles[ret - 1][0], circles[ret - 1][1], circles[ret - 1][2]
        else:
            return False, 0, 0, 0
    else:
        return False, 0, 0, 0


def add_circle_panel(key_code, v_id, v_name, min_radius, max_radius):
    file = '3/' + v_name + '.png'
    b, x, y, r = detect_circle_panel(file, min_radius, max_radius, v_name)
    if not b:
        return
    xc, yc = x, y
    save_circle_panel(file, x, y, r, '4/' + v_name + '.png')
    save_panel_axis(key_code, v_id, v_name, x - r, y - r, x + r, y + r, 0)

    axis = read_panel_axis()

    for d in axis:
        if d['name'] == v_name:
            found, x, y = sift_match_zone(file, v_name, round(float(d['left'])) - 2, round(float(d['top'])) - 2,
                                          round(float(d['right'])) + 2, round(float(d['bottom'])) + 2)
            if found:
                print(v_name, 'found at', (x, y), 'center offset is', (x - xc, y - yc))


def fake():
    pass
    # edges = cv2.Canny(gray,50,200)
    # imS = cv2.resize(edges, (800, 400))
    # cv2.imshow('edges', imS)
    # cv2.waitKey(100)
    # img = cv2.imread('off.png')
    # img = cv2.imread('3/armor_off.png')
    # img = cv2.imread('3/chopper.png')
    # img = img[0:40,0:175]
    # img = img[460:520,580:710]

    # rect = (8, 13, 97, 37)
    # rect = (580, 460, 120, 50)
    # rect = (0, 5, 173, 41)


def detect_rectangle_contour(name, x, y, w, h):
    img = cv2.imread(name)
    img = img[y:y + h, x:x + w]

    # median = cv2.medianBlur(img, 15)
    # img_rsz = cv2.resize(median, (800, 400))
    # cv2.imshow('median', img_rsz)
    # cv2.waitKey(100)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    img_rsz = cv2.resize(erosion, (800, 400))
    cv2.imshow('erosion', img_rsz)
    cv2.waitKey(100)

    # dilation = cv2.dilate(img, kernel, iterations=1)
    # img_rsz = cv2.resize(dilation, (800, 400))
    # cv2.imshow('dilation', img_rsz)
    # cv2.waitKey(100)

    gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    img_rsz = cv2.resize(gray, (800, 400))
    cv2.imshow('gray', img_rsz)
    cv2.waitKey(100)

    # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY_INV, 11, 2)

    img_rsz = cv2.resize(thresh, (800, 400))
    cv2.imshow('thresh', img_rsz)
    cv2.waitKey(100)

    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    img_rsz = cv2.resize(closing, (800, 400))
    cv2.imshow('closing', img_rsz)
    cv2.waitKey(100)

    contours, hierarchy = cv2.findContours(closing, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10:
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

                print(x, y, w, h)

    img_rsz = cv2.resize(img, (800, 400))
    cv2.imshow(name, img_rsz)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_rectangle_grab_cut(name, left, top, right, bottom):
    origin = cv2.imread(name)
    img = origin[top:bottom, left:right]
    mask = np.zeros(img.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1] - 1, img.shape[0] - 1)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    img_rsz = cv2.resize(img, (800, 400))
    cv2.imshow('grab_cut', img_rsz)
    cv2.waitKey(100)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rsz = cv2.resize(gray, (800, 400))
    cv2.imshow('gray', img_rsz)
    cv2.waitKey(100)

    kernel = np.ones((6, 6), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    img_rsz = cv2.resize(closing, (800, 400))
    cv2.imshow('closing', img_rsz)
    cv2.waitKey(100)

    thresh = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)

    img_rsz = cv2.resize(thresh, (800, 400))
    cv2.imshow('thresh', img_rsz)
    cv2.waitKey(100)

    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10:
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                print(x, y, w, h)

    img_rsz = cv2.resize(img, (800, 400))
    cv2.imshow('image', img_rsz)
    cv2.waitKey()


def detect_supply_box():
    # img = cv2.imread('1/select_car.png')#120
    img = cv2.imread('3/custom_supply.png')
    # img = cv2.imread('3/system_supply.png')
    # img = cv2.imread('3/system_supply_1.png')
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


def image_folder_resize():
    lst = os.listdir('1')
    for name in lst:
        print(name)
        resize_image(name)


def resize_image(name):
    image = cv2.imread('1/' + name)
    resized = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('3/' + name, resized)


def save_screenshot(name):
    # show_full_screen('1/'+name)
    img2 = pyautogui.screenshot(region=(0, 0, 1280, 720))
    img2.save('3/' + name)


def byte_copy():
    data = [ctypes.c_byte()] * 4
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
    img = cv2.imread('save.png', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 120, 255, 0)
    cv2.imshow('img', thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cut_image(folder, name):
    img = cv2.imread(folder + name)
    crop = img[0:720, 0:1280]
    cv2.imwrite(folder + name, crop)


def find_homo():
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    MIN_MATCH_COUNT = 10
    img1 = cv.imread('4/print_vehicle.png', cv.IMREAD_GRAYSCALE)  # queryImage
    # img1 = cv.imread('4/print_vehicle.png', cv.IMREAD_GRAYSCALE)  # queryImage
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
    print('good:', len(good), len(kp1), len(kp2), len(matches), len(good) / len(matches))
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()


def paste_image_folder():
    files = os.listdir('3')
    for f in files:
        img = cv2.imread('3/' + f)
        resized = cv2.resize(img, (1213, 682), interpolation=cv2.INTER_LINEAR)
        img2 = numpy.zeros((720, 1280, 3), numpy.uint8)
        img2[38:720, 32:1245] = resized
        cv2.imwrite('5/' + f, img2)


def paste_image_file(f):
    img = cv2.imread('3/' + f)
    resized = cv2.resize(img, (1213, 682), interpolation=cv2.INTER_LINEAR)
    img2 = numpy.zeros((720, 1280, 3), numpy.uint8)
    img2[38:720, 32:1245] = resized
    cv2.imwrite('5/' + f, img2)


def add_panels():
    add_circle_panel(33, 0, 'take_drive', 40, 44)
    add_circle_panel(33, 1, 'door', 25, 29)
    add_circle_panel(33, 2, 'strop_on', 35, 39)
    add_circle_panel(33, 3, 'strop_off', 35, 39)
    add_circle_panel(33, 4, 'tough_on', 35, 39)
    add_circle_panel(33, 5, 'tough_off', 35, 39)
    add_circle_panel(33, 6, 'print_vehicle', 20, 35)
    add_circle_panel(33, 7, 'update_chip', 20, 35)
    add_circle_panel(33, 8, 'select_weapon', 36, 40)
    add_circle_panel(33, 9, 'para', 36, 50)
    add_circle_panel(33, 10, 'hacker', 35, 39)
    add_rectangle_panel(33, 11, 'pick_hand_yellow', 46, 46)
    add_rectangle_panel(33, 12, 'pick_hand_white', 46, 46)
    add_circle_panel(34, 0, 'drive_by', 40, 44)
    add_rectangle_panel(34, 1, 'pick_box_yellow', 46, 46)
    add_rectangle_panel(34, 2, 'pick_box_white', 46, 46)
    add_circle_panel(58, 0, 'ex_seat', 32, 36)


def ocr():
    import pytesseract as tess
    print(tess.get_tesseract_version())
    print(tess.get_languages())

    # image = cv2.imread("ocr.png")
    image = cv2.imread('chinese.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    start = time.time()
    # text = tess.image_to_string(image_rgb, lang="eng")
    text = tess.image_to_string(thresh, lang="chi_sim")
    print(time.time() - start, text)
    content = text.replace("\f", "").split("\n")
    for c in content:
        if len(c) > 0:
            print(c)
    h, w, c = image.shape
    boxes = tess.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    cv2.imshow('text detect', image)
    cv2.waitKey(3)
    cv2.destroyAllWindows()


def add_rectangle_panel(v_key_code, v_id, v_name, width, height):
    file = '3/' + v_name + '.png'
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    lst_rect = []
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(width - w) <= 3 and abs(height - h) <= 3:
                idx += 1
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(img, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(x, y, w, h)
                lst_rect.append((x, y, w, h))

    cv2.imshow(v_name, img)
    ret = cv2.waitKey(0)
    ret = int(ret - 0x30)
    cv2.destroyAllWindows()

    if ret <= 0:
        return

    ret -= 1
    left, top, right, bottom = lst_rect[ret][0], lst_rect[ret][1], lst_rect[ret][0] + lst_rect[ret][2], lst_rect[ret][
        1] + lst_rect[ret][3]
    panel = cv2.imread(file)
    panel = panel[top:bottom, left:right]
    cv2.imwrite('4/' + v_name + '.png', panel)

    save_panel_axis(v_key_code, v_id, v_name, left, top, right, bottom, 0)
    axis = read_panel_axis()

    for d in axis:
        if d['name'] == v_name:
            found, x, y = sift_match_zone(file, v_name, round(float(d['left'])) - 2, round(float(d['top'])) - 2,
                                          round(float(d['right'])) + 2, round(float(d['bottom'])) + 2)
            if found:
                xc = (left + right) / 2
                yc = (top + bottom) / 2
                print(v_name, 'found at', (x, y), 'center offset is', (x - xc, y - yc))



# sift_match('4/armor_off.png', '3/armor_off.png')

# detect_rectangle('6/armor_off.png', 0, 0, 175, 40)
# detect_rectangle('6/drive.png', 0, 0, 175, 40)
# detect_rectangle('6/replace.png', 0, 0, 175, 40)
# grab_cut('3/armor_off.png', 580, 460, 710,  520)
# grab_cut('3/select_vehicle.png',536, 524, 736, 560)
# find_homo()
cut_image('3/', 'print_weapon.png')


# detect_circle('3/lean_out.png')
# test4('moto.png')
# sift_match()
# find_homo()

# byte_copy()
# folder_resize()
# resize_image('redbox.png')
# update_screen()
# detect_supply_box()
# detect_random_supply()
# test_dict()
# sift_match_zone('1/redbox.png', 'suply', 392, 302, 430, 600)
# sift_match_zone('1/redbox.png', 'suply', 392, 440, 430, 600)
# if __name__ == '__main__':
if __name__ == '__':
    test7()
    detect_door()
    detect_drive()
    detect_tough_guy()
