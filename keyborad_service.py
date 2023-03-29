import threading

import keyboard

import tcp_service
from dectect_mode import detect_sub_mode
from defs import TcpData, EventType, KeyEvent, set_param1, set_param2, ControlEvent, SubModeType, MapModeStatus, \
    MainModeType, TransPointStatus
from window_info import window_info_init

current_sub_mode = SubModeType.NONE_SUB_MODE

def key_event_callback(evt):
    data = TcpData()
    data.type = EventType.TYPE_KEYBOARD
    set_param1(data, evt.scan_code)
    if evt.event_type == 'down':
        set_param2(data, KeyEvent.KEY_DOWN)
    elif evt.event_type == 'up':
        set_param2(data, KeyEvent.KEY_UP)

    tcp_service.tcp_data_append(data)
    print(evt.name, evt.scan_code, evt.event_type)

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


def sub_mode_switch():

    if mode_info.sub_mode == SubModeType.NONE_SUB_MODE:
        i = detect_sub_mode()
        if i >= 0:
            mode_info.sub_mode = i + SubModeType.SUB_MODE_OFFSET
            print("switch to sub mode", i)
            data = TcpData()
            data.type = EventType.TYPE_CONTROL
            set_param1(data, ControlEvent.SUB_MODE)
            set_param2(data, i + mode_info.sub_mode)
            tcp_service.tcp_data_append(data)
        else:
            print("no sub mode detected")
    else:
        mode_info.sub_mode = SubModeType.NONE_SUB_MODE
        data = TcpData()
        data.type = EventType.TYPE_CONTROL
        set_param1(data, ControlEvent.SUB_MODE)
        set_param2(data, SubModeType.NONE_SUB_MODE)
        tcp_service.tcp_data_append(data)


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


def thread_key():
    keyboard.hook(key_event_callback)
    keyboard.add_hotkey('alt+f1', main_mode_switch, args=(str(MainModeType.MULTI_PLAYER)))
    keyboard.add_hotkey('alt+f2', main_mode_switch, args=(str(MainModeType.BATTLE_GROUND)))
    keyboard.add_hotkey('alt+f3', main_mode_switch, args=(str(MainModeType.PVE)))
    keyboard.add_hotkey('f', sub_mode_switch)
    keyboard.add_hotkey('m', map_mode_switch)
    keyboard.add_hotkey('ctrl', trans_point_mode_switch)
    keyboard.wait()


class KeyboardService:
    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
