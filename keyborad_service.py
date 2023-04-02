import threading

import keyboard

import tcp_service

from defs import TcpData, EventType, KeyEvent, set_param1, set_param2, ControlEvent, SubModeType, MapModeStatus, \
    MainModeType, TransPointStatus

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

def thread_key():
    keyboard.hook(key_event_callback)
    keyboard.add_hotkey('alt+f1', main_mode_switch, args=(str(MainModeType.MULTI_PLAYER)))
    keyboard.add_hotkey('alt+f2', main_mode_switch, args=(str(MainModeType.BATTLE_GROUND)))
    keyboard.add_hotkey('alt+f3', main_mode_switch, args=(str(MainModeType.PVE)))
    keyboard.add_hotkey('l', sub_mode_switch)
    keyboard.add_hotkey('c', map_mode_switch)
    keyboard.add_hotkey('enter', trans_point_mode_switch)
    keyboard.wait()


class KeyboardService:
    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
