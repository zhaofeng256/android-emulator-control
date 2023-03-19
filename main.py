from tcp_service import TcpServerService
from keyborad_service import KeyboardService
from mouse_service import MouseService
from time import sleep
import signal
import sys

def signal_handler(signal, frame):
    print('main exit')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    tcp_svc = TcpServerService('', 65432)
    ret = tcp_svc.start()
    if ret == 0:
        print('start tcp server successfully')

    kbd_svc = KeyboardService()
    kbd_svc.start()



    ms_svc = MouseService()
    ms_svc.start()

    while True:
        sleep(1)