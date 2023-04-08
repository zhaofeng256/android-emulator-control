import ctypes
import threading

import win32gui
from win32con import SW_SHOWNOACTIVATE

import tcp_service
from defs import TcpData, set_param1_int32, EventType, SettingType


class WindowInfo:
    window_size = [1280, 720]
    window_pos = [0, 0]
    main_title = '腾讯手游助手(64位)'
    sub_title = 'sub'
    main_hwnd = 0
    terminal_size = [1280, 720]
    condition = threading.Condition()
    lock = threading.Lock()


def main_callback(hwnd, extra):
    title = win32gui.GetWindowText(hwnd)
    if title == extra.main_title:
        # print(title)
        with WindowInfo.lock:
            WindowInfo.main_hwnd = hwnd
        # with extra.condition:
        #     extra.condition.notify_all()
    return 1


def sub_callback(hwnd, extra):
    rect = win32gui.GetWindowRect(hwnd)
    title = win32gui.GetWindowText(hwnd)
    # if len(title) > 0:
    #     print("child" , title, rect)
    if title == extra.sub_title:
        with WindowInfo.lock:
            WindowInfo.window_pos = rect[0], rect[1]
            WindowInfo.window_size = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
            # WindowInfo.window_pos = 0,0
            # WindowInfo.window_size = 1280,720
            print('window pos:', WindowInfo.window_pos, 'size', WindowInfo.window_size)
        # with extra.condition:
        #     extra.condition.notify_all()
    return 1


def get_window_pos_size():

    win32gui.EnumWindows(main_callback, WindowInfo)

    with WindowInfo.lock:
        hwnd = WindowInfo.main_hwnd

    win32gui.EnumChildWindows(hwnd, sub_callback, WindowInfo)



def update_window_info():
    get_window_pos_size()
    with WindowInfo.lock:
        left = WindowInfo.window_pos[0]
        hwnd = WindowInfo.main_hwnd

    # show window if it is minimized
    if left < 0:
        win32gui.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
        get_window_pos_size()


def get_window_info():
    with WindowInfo.lock:
        window_pos = WindowInfo.window_pos
        window_size = WindowInfo.window_size

    return window_pos[0], window_pos[1], window_size[0], window_size[1]


def send_window_info():
    with WindowInfo.lock:
        window_pos = WindowInfo.window_pos
        window_size = WindowInfo.window_size

    data = TcpData()
    # send window position
    data.type = EventType.TYPE_SET_WINDOW
    set_param1_int32(data, SettingType.WINDOW_POS)
    a = ctypes.c_int16(window_pos[0])
    b = ctypes.c_int16(window_pos[1])
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset), ctypes.byref(a), 2)
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset + 2), ctypes.byref(b), 2)
    tcp_service.tcp_data_append(data)
    # send window size
    set_param1_int32(data, SettingType.WINDOW_SIZE)
    a = ctypes.c_int16(window_size[0])
    b = ctypes.c_int16(window_size[1])
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset), ctypes.byref(a), 2)
    ctypes.memmove(ctypes.byref(data, TcpData.param2.offset + 2), ctypes.byref(b), 2)
    tcp_service.tcp_data_append(data)


def get_screen_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def move_window(x1, y1, x2, y2):
    with WindowInfo.lock:
        main_hwnd = WindowInfo.main_hwnd
    ctypes.windll.user32.MoveWindow(main_hwnd, x1, y1, x2, y2, True)


def get_terminal_size():
    with WindowInfo.lock:
        width = WindowInfo.terminal_size[0]
        height = WindowInfo.terminal_size[1]
    return width, height


def set_terminal_size(width, height):
    with WindowInfo.lock:
        WindowInfo.terminal_size[0] = width
        WindowInfo.terminal_size[1] = height
        print("terminal size",(width, height))


def get_relative_position(x, y):
    with WindowInfo.lock:
        emulator_resolution = WindowInfo.terminal_size
        window_pos = WindowInfo.window_pos
        window_size = WindowInfo.window_size

    return round(emulator_resolution[0] * (x - window_pos[0]) / window_size[0]), \
        round(emulator_resolution[1] * (y - window_pos[1]) / window_size[1])

# shell = win32com.client.Dispatch("WScript.Shell")
# shell.SendKeys('%')
# win32gui.SetForegroundWindow(window_hwnd)
# SW_SHOW, SW_NORMAL, SW_MAXIMIZE, SW_SHOWMINNOACTIVE, SW_FORCEMINIMIZE


# def main():
#     update_window_info()
#     get_window_info()
#
#
# if __name__ == '__main__':
#     main()
