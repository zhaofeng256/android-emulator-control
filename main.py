import signal
import sys
from time import sleep

import defs
from keyborad_service import KeyboardService
from mouse_service import MouseService
from tcp_service import TcpServerService


def signal_handler(a,b):
    print('main exit')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    TcpServerService('', defs.TCP_PORT).start()
    MouseService.start()
    KeyboardService.start()

    while True:
        sleep(1)
