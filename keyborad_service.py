import threading
import keyboard
import tcp_service


TYPE_KEYBOARD = 0
TYPE_MOUSE = 1
KEY_DOWN = 0
KEY_UP = 1

import time
def k_down(e):
    print(e.name, e.event_type)

def k_up(e):
    print(e.name, e.event_type)

# keyboard.on_press_key('k', k_down, suppress=False)
# keyboard.on_release_key('k', k_up, suppress=False)



class KeyboardService(object):
    def start(self):
        k_thread = threading.Thread(target=self.thread_key, args=())
        k_thread.daemon = True
        k_thread.start()
    def thread_key(self):
        keyboard.hook(print_pressed_keys)
        keyboard.wait()

def print_pressed_keys(evt):
        data = tcp_service.TcpData()
        data.type = TYPE_KEYBOARD
        data.param1 = evt.scan_code
        if evt.event_type == 'down':
            data.param2 = KEY_DOWN
        else:
            data.param2 = KEY_UP
        tcp_service.tcp_data_append(data)
        print(evt.name, evt.event_type)


