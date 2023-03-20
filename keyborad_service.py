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
    print(evt.name, evt.event_type)


def thread_key():
    keyboard.hook(key_event_callback)
    keyboard.wait()


class KeyboardService:
    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
