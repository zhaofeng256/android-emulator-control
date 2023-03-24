import threading
import time

import mouse

import tcp_service
from defs import EventType, TcpData, set_param1, set_param2, ButtonType, KeyEvent, WheelEvent


def mouse_event_callback(evt):
    t = type(evt)
    data = TcpData()

    if t == mouse._mouse_event.MoveEvent:
        if MouseService.stop_move:
            return
        data.type = EventType.TYPE_MOUSE
        set_param1(data, evt.x)
        set_param2(data, evt.y)
        #print('move', evt.x, evt.y, time.time())
    elif t == mouse._mouse_event.ButtonEvent:
        data.type = EventType.TYPE_BUTTON
        if evt.button == 'left':
            set_param1(data, ButtonType.LEFT)
        elif evt.button == 'right':
            set_param1(data, ButtonType.RIGHT)
        elif evt.button == 'middle':
            set_param1(data, ButtonType.MIDDLE)
        elif evt.button == 'x':
            set_param1(data, ButtonType.BACK)
        elif evt.button == 'x2':
            set_param1(data, ButtonType.FORWARD)
        if evt.event_type == 'down':
            set_param2(data, KeyEvent.KEY_DOWN)
        elif evt.event_type == 'up':
            set_param2(data, KeyEvent.KEY_UP)
        #print('button', evt.button, evt.event_type)
    elif t == mouse._mouse_event.WheelEvent:
        if MouseService.stop_move:
             return
        data.type = EventType.TYPE_WHEEL
        if evt.delta == -1.0:
            set_param1(data, WheelEvent.ROLL_BACK)
            #print('roll back')
        elif evt.delta == 1.0:
            set_param1(data, WheelEvent.ROLL_FORWARD)
            #print('roll forward')
        set_param2(data, 0)
    else:
        return

    tcp_service.tcp_data_append(data)


def thread_mouse():
    mouse.hook(mouse_event_callback)
    mouse.wait()


class MouseService:
    stop_move = True

    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_mouse, args=())
        k_thread.daemon = True
        k_thread.start()
