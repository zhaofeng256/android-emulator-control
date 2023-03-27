import time
import win32gui, win32ui, win32con, win32api


import pyautogui
import cv2
import numpy as np

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

#windows screenshot
def test():
    beg = time.time()
    window_capture(1000, 500, 280, 220, "haha.jpg")
    end = time.time()
    print(end - beg)

#pyautogui screenshot
def test1(name):
    beg = time.time()
    img = pyautogui.screenshot(region=[870, 350, 1280-870, 720-350])
    end = time.time()
    img.save(name);
    print(end - beg)

#screenshot find axis of picture file
def test2():
    beg = time.time()
    pos = pyautogui.locateOnScreen('coyote1.png')
    end = time.time()
    print(end - beg)
    print(pos)

    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # cv2.imshow("截屏",img)
    # cv2.waitKey(0)

#compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse, diff

#opencv screenshot compare with picture file
def test3(name):
    # load the input images
    img1 = cv2.imread(name)
    img2 = cv2.imread('coyote.png')

    beg = time.time()
    img3 = pyautogui.screenshot(region=[870, 350, 1280-870, 720-350])
    # convert the images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray'+name,img3)

    error, diff = mse(img1, img3)
    end = time.time()
    print('last', end - beg)

    print("Image matching Error between the two images:",error)

import cv2
import numpy as np

def test4(name):
    image = cv2.imread(name)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.26, minDist=80,
        param1=60, param2=40, minRadius=38, maxRadius=45)
    # If some circle is found
    if circles is not None:
       # Get the (x, y, r) as integers
       circles = np.round(circles[0, :]).astype("int")
       print(circles)
       # loop over the circles
       for (x, y, r) in circles:
          cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("circle",output)
    cv2.waitKey(0)


def test5(name):

    #use canny, as HoughCircles seems to prefer ring like circles to filled ones.
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
    #smooth to reduce noise a bit more
    #cv2.Smooth(img, img, cv.CV_GAUSSIAN, 7, 7)

name = "test.png"
#test1(name)
#test3(name)
test4('shotpng')