
from tcp_service import TcpServerService
from time import sleep
from keyborad_service import KeyboardService

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

    while True:
        sleep(1)