import threading

import mouse

import tcp_service
from defs import EventType, TcpData, set_param1, set_param2, ButtonType, KeyEvent, WheelEvent, MapModeStatus, \
    TransPointStatus
from switch_mode import ModeInfo
from window_info import get_relative_position


def mouse_event_callback(evt):
    t = type(evt)
    data = TcpData()

    if t == mouse._mouse_event.MoveEvent:
        # unuse frequent move because event report per 8ms while emulator move takes 17ms
        if evt.time < MouseService.mouse_timeout:
            return
        MouseService.mouse_axis_x = evt.x
        MouseService.mouse_axis_y = evt.y
        MouseService.mouse_timeout = evt.time + 0.01
        if MouseService.stop_move:
            return
        data.type = EventType.TYPE_MOUSE_AXIS

        # show axis at status bar
        if ModeInfo.map_mode_on == MapModeStatus.MAP_MODE_ON or \
                ModeInfo.transparent_mode_on == TransPointStatus.TRANSPARENT_ON:
            x, y = get_relative_position(MouseService.mouse_axis_x, MouseService.mouse_axis_y)
        else:
            x, y = evt.x, evt.y
        MouseService.statusbar.showMessage(str(x) + ',' + str(y))
        # print('move', evt.x, evt.y, evt.time)

        # send real mouse axis
        set_param1(data, evt.x)
        set_param2(data, evt.y)

    elif t == mouse._mouse_event.ButtonEvent:
        data.type = EventType.TYPE_MOUSE_BUTTON
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
        if evt.event_type == 'down' or evt.event_type == 'double':
            set_param2(data, KeyEvent.KEY_DOWN)
        elif evt.event_type == 'up':
            set_param2(data, KeyEvent.KEY_UP)
        # print('button', evt.button, evt.event_type)
    elif t == mouse._mouse_event.WheelEvent:
        if MouseService.stop_move:
            return
        data.type = EventType.TYPE_MOUSE_WHEEL
        if evt.delta == -1.0:
            set_param1(data, WheelEvent.ROLL_BACK)
            # print('roll back')
        elif evt.delta == 1.0:
            set_param1(data, WheelEvent.ROLL_FORWARD)
            # print('roll forward')
        set_param2(data, 0)
    else:
        return

    tcp_service.tcp_data_append(data)


def thread_mouse():
    mouse.hook(mouse_event_callback)
    mouse.wait()


class MouseService:
    stop_move = False
    mouse_axis_x = 0
    mouse_axis_y = 0
    mouse_timeout = 0

    @staticmethod
    def start():
        k_thread = threading.Thread(target=thread_mouse, args=())
        k_thread.daemon = True
        k_thread.start()

    def set_statusbar(s):
        if not hasattr(MouseService, 'statusbar'):
            MouseService.statusbar = s
