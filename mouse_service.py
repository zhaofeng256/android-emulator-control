import threading
import mouse
import tcp_service
import defs

def mouse_event_callback(evt):

    t = type(evt)
    data = tcp_service.TcpData()

    if t == mouse._mouse_event.MoveEvent:
        data.type = defs.EventType.TYPE_MOUSE
        data.param1 = evt.x
        data.param2 = evt.y
        #print('move',evt.x, evt.y)
    elif t == mouse._mouse_event.ButtonEvent:
        data.type = defs.EventType.TYPE_BUTTON
        if evt.button == 'left':
            data.param1 = defs.ButtonType.LEFT
        elif evt.button == 'right':
            data.param1 = defs.ButtonType.RIGHT
        elif evt.button == 'middle':
            data.param1 = defs.ButtonType.MIDDLE
        elif evt.button == 'x':
            data.param1 = defs.ButtonType.BACK
        elif evt.button == 'x2':
            data.param1 = defs.ButtonType.FORWARD
        if evt.event_type == 'down':
            data.param2 = defs.KeyEvent.KEY_DOWN
        elif evt.event_type == 'up':
            data.param2 = defs.KeyEvent.KEY_UP

        print(evt.button, evt.event_type)
    elif t == mouse._mouse_event.WheelEvent:
        data.type = defs.EventType.TYPE_WHEEL
        if evt.delta == -1.0:
            data.param1 = defs.WheelEvent.ROLL_BACK
            print('roll back')
        elif evt.delta == 1.0:
            data.param1 = defs.WheelEvent.ROLL_FORWARD
            print('roll forward')
        data.param2 = 0

    tcp_service.tcp_data_append(data)

class MouseService():
    def start(self):
        k_thread = threading.Thread(target=self.thread_mouse, args=())
        k_thread.daemon = True
        k_thread.start()
    def thread_mouse(self):
        mouse.hook(mouse_event_callback)
        mouse.wait()

