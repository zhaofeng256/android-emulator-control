import threading

import keyboard

import tcp_service
from defs import TcpData, EventType, KeyEvent, set_param1, set_param2


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

def mode_switch(mode):
    print("switch to mode", mode)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1(data, 0)
    set_param2(data, int(mode))
    tcp_service.tcp_data_append(data)

def thread_key():
    keyboard.hook(key_event_callback)
    keyboard.add_hotkey('ctrl+f1', mode_switch, args=('0'))
    keyboard.add_hotkey('ctrl+f2', mode_switch, args=('1'))
    keyboard.add_hotkey('ctrl+f3', mode_switch, args=('2'))
    keyboard.wait()


class KeyboardService:
    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
