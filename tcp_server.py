import socket
import signal
import sys
import threading
from os import error
from time import sleep


def signal_handler(signal, frame):
    print('exit')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
def test_tcp_server():

    s = socket.socket()
    host_name = socket.gethostname()
    host = socket.gethostbyname(host_name)
    print('host ip:' + host)
    port = 65432
    s.bind((host, port))
    s.listen(5)
    c, addr = s.accept()
    print('accepted: %s' % (addr,))
    c.send(str.encode('hello this is server'))

    while True:
        msg = c.recv(1024)
        print('recv :', msg)


class TcpServerService(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            lst = threading.Thread(target=self.thread_listen, args=())
            lst.daemon = True
            lst.start()
            return 0
        except:
            print('start tcp server error')
            return 1

    def thread_listen(self):
        while True:
            self.sock.listen(5)
            client, addr = self.sock.accept()
            print('accepted: %s' % (addr,))
            client.settimeout(60)
            snd = threading.Thread(target=self.thread_send, args=(client, addr))
            snd.daemon = True
            snd.start()

    def thread_send(self, client, address):
        while True:
            msg = 'hello this is server\n'
            try:
                sleep(1)
                print('send hello')
                client.send(str.encode(msg))
                data = client.recv(1024)
                if data:
                    print('recv:', data)
                else:
                    raise error('client disconnected')
            except:
                print('client close')
                client.close()
                return False

if __name__ == "__main__":

    svc = TcpServerService('', 65432)
    ret = svc.start()
    if ret == 0:
        print('start tcp server successfully')
    while True:
        sleep(1)


