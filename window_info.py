import ctypes, sys

import win32gui, win32com.client
from pywinauto import keyboard
from win32con import SW_SHOWNOACTIVATE


def callback(hwnd, extra):
    rect = win32gui.GetWindowRect(hwnd)
    title = win32gui.GetWindowText(hwnd)
    # if len(title) > 0:
    #     print(title)
    if title == extra.window_title:
        print(title)
        extra.window_pos = rect[0], rect[1]
        extra.window_size = rect[2] - rect[0], rect[3] - rect[1]
        extra.window_hwnd = hwnd


def get_screen_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class WindowInfo():
    def __init__(self):
        self.emulator_resolution = [0, 0]
        self.window_hwnd = 0
        self.window_size = [1280, 720]
        self.window_pos = [0, 0]
        self.window_title = ''

    def get_window_pos_size(self, title):
        self.window_title = title
        win32gui.EnumWindows(callback, self)

    def get_window_pos(self):
        return self.window_pos

    def get_window_size(self):
        return self.window_size

    def get_emulator_resolution(self):
        self.emulator_resolution = [1280, 720]
        return self.emulator_resolution


    def get_relative_position(self, x, y):
        return round(self.emulator_resolution[0] * (x - self.window_pos[0]) / self.window_size[0]), \
            round(self.emulator_resolution[1] * (y - self.window_pos[1]) / self.window_size[1])


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


# shell = win32com.client.Dispatch("WScript.Shell")
# shell.SendKeys('%')
# win32gui.SetForegroundWindow(window_hwnd)
# SW_SHOW, SW_NORMAL, SW_MAXIMIZE, SW_SHOWMINNOACTIVE, SW_FORCEMINIMIZE
info = WindowInfo()
def window_info_init():
    global info
    info.get_window_pos_size('腾讯手游助手(64位)')

    if info.window_pos[0] == -4:
        # full screen
        # keyboard.press('f11');
        print('fail to get window size')
        window_pos = 0, 0
        window_size = get_screen_resolution()
    elif info.window_pos[0] < 0:
        # minimized
        print('window pos:', info.window_pos, 'size', info.window_size, info.window_hwnd)
        win32gui.ShowWindow(info.window_hwnd, SW_SHOWNOACTIVATE)
        info.get_window_pos_size('腾讯手游助手(64位)')

    print('window pos:', info.window_pos, 'size', info.window_size, info.window_hwnd)
    emulator_resolution = info.get_emulator_resolution()
    return info

# def main():
#     x, y = get_relative_position(1000, 500)
#     print(x, y)
#
#
# if __name__ == '__main__':
#     main()
