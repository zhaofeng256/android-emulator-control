import threading

import keyboard

import switch_mode
import tcp_service
from defs import TcpData, EventType, KeyEvent, set_param1, set_param2, SubModeType, MainModeType

current_sub_mode = SubModeType.NONE_SUB_MODE

dc_kbd = {}
def key_event_callback(evt):
    # filter same press
    global dc_kbd
    if evt.name in dc_kbd.keys():
        if dc_kbd[evt.name] == evt.event_type:
            return
    dc_kbd[evt.name] = evt.event_type

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
    keyboard.add_hotkey('alt+f1', switch_mode.main_mode_switch, args=(str(MainModeType.MULTI_PLAYER)))
    keyboard.add_hotkey('alt+f2', switch_mode.main_mode_switch, args=(str(MainModeType.BATTLE_GROUND)))
    keyboard.add_hotkey('alt+f3', switch_mode.main_mode_switch, args=(str(MainModeType.PVE)))
    keyboard.add_hotkey('c', switch_mode.map_mode_switch)
    keyboard.add_hotkey('right shift', switch_mode.trans_point_mode_switch)
    keyboard.wait()


class KeyboardService:
    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
