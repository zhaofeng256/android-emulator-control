import ctypes
import socket
import signal
import sys
import threading
from os import error
from time import sleep


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

tcp_data_list = []
tcp_list_cv = threading.Condition()
tcp_data_id = 0
class TcpData(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("type", ctypes.c_char),
        ("param1", ctypes.c_int),
        ("param2", ctypes.c_int),
    ]

def tcp_data_append(data):

    with tcp_list_cv:
        data.id = tcp_data_id
        tcp_data_list.append(data)
        tcp_list_cv.notify_all()
def tcp_data_pop():
    with tcp_list_cv:
        while not len(tcp_data_list):
            tcp_list_cv.wait()
        return tcp_data_list.pop()

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
            send_data = tcp_data_pop();

            print('send',send_data.id, send_data.type, send_data.param1, send_data.param2)

            try:
                client.send(bytearray(send_data))
                recv_data = client.recv(1024)
                if recv_data:
                    print('recv:', recv_data)
                else:
                    raise error('client disconnected')
            except Exception as e:
                print(e)
                client.close()
                return False


