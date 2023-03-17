from time import sleep

import win32gui
import win32con
import win32ui
import win32api
import time
import pywinauto

#from sendinput import *

import keyboard
def callback(handle, param):
    s = win32gui.GetClassName(handle)
    print(s)
    #if s == 'ScrollBar':
    if s == 'SysTabControl32':
        try:
            print(f'Sending key to {handle}, {s}')

            # win32gui.SendMessage(handle, win32con.WM_KEYDOWN, win32con.VK_NEXT, 0)
            # win32gui.SendMessage(handle, win32con.WM_KEYUP, win32con.VK_NEXT, 0)
            win32gui.SendMessage(handle, win32con.WM_KEYDOWN, win32con.VK_RBUTTON, 0)
            win32gui.SendMessage(handle, win32con.WM_KEYUP, win32con.VK_RBUTTON, 0)
            sleep(1)
        except Exception:
            print('Exception sending to {handle}, {s}')

def test():
    #Sending key to 8653822, AEngineRenderWindowClass
    #Sending key to 4131814, subWin
    #hwnd = win32gui.FindWindow(None, "DZ")
    hwnd = win32gui.FindWindow(None, '*new 4 - Notepad++')

    # win = win32ui.CreateWindowFromHandle(hwnd)
    #
    # while True:
    #     print("Sending")
    #     win.SendMessage(win32con.WM_CHAR, ord('W'), 0)
    #     sleep(2)

    # for i in range(10):
    #     win32gui.PostMessage(0x00C70468, win32con.WM_KEYDOWN, win32con.VK_RBUTTON, 0)
    #     win32gui.PostMessage(0x00C70468, win32con.WM_KEYUP, win32con.VK_RBUTTON, 0)
    #     sleep(2)
    win32gui.EnumChildWindows(hwnd, callback, 0)


def test_winauto():
    app = pywinauto.Application(backend="win43").start(r"C:\Program Files (x86)\Notepad++\notepad++.exe")

    time.sleep(3)
    print('sleeping over')
    Wizard = app['*new 4 - Notepad++']

    while True:
        Wizard.send_keystrokes("{VK_RETURN}")
        time.sleep(1)


def cb_tcgame(handle, param):
    s = win32gui.GetClassName(handle)
    print(s)
    # if s == 'ScrollBar':
    #if s == 'subWin':
    try:
        print(f'Sending key to {handle}, {s}')
        for i in range(3):
            win32api.keybd_event(0x57, 0, )

    except Exception:
        print('Exception sending to {handle}, {s}')

def test_tcgame():
    hwnd = win32gui.FindWindow(None, '腾讯手游助手(64位)')
    win32gui.SetForegroundWindow(hwnd)
    win32gui.EnumChildWindows(hwnd, cb_tcgame, 0)

def cb_notepad(handle, param):
    s = win32gui.GetClassName(handle)
    print(s)
    if s == 'SysTabControl32':
        try:
            print(f'Sending key to {handle}, {s}')

            win32api.keybd_event(0x46, 0, )
            send_unicode('w')
            win32gui.SendMessage(handle, win32con.WM_CHAR, 0x57, 0)
            win32gui.SendMessage(handle, win32con.WM_CHAR, 0x57, 0)
        except Exception:
            print('Exception sending to {handle}, {s}')

def test_notepad():

    hwnd = win32gui.FindWindow(None, '*new 4 - Notepad++')
    win32gui.SetForegroundWindow(hwnd)
    win32gui.EnumChildWindows(hwnd, cb_notepad, 0)

def test_keyboard():
    for i in range(20):
        keyboard.press('a')
        time.sleep(1)
    keyboard.release('a')
    # keyboard.wait('space')
    # print('space was pressed, continuing...')
    # while(True):
    #     keyboard.send(47)
    #     sleep(1)v

test_keyboard()
#test_tcgame()
#test_notepad()