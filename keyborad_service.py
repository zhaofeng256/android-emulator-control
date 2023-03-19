import threading
import keyboard
import tcp_service
import defs

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
        keyboard.hook(key_event_callback)
        keyboard.wait()

def key_event_callback(evt):
        data = tcp_service.TcpData()
        data.type = defs.EventType.TYPE_KEYBOARD
        data.param1 = evt.scan_code
        if evt.event_type == 'down':
            data.param2 = defs.KeyEvent.KEY_DOWN
        elif evt.event_type == 'up':
            data.param2 = defs.KeyEvent.KEY_UP

        tcp_service.tcp_data_append(data)
        print(evt.name, evt.event_type)


