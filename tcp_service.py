import socket
import struct
import threading
from os import error
from ctypes import sizeof

tcp_data_list = []
tcp_list_cv = threading.Condition()

from defs import *


def tcp_data_append(data):
    with tcp_list_cv:
        TcpServerService.tcp_data_id += 1
        set_id(data, TcpServerService.tcp_data_id)
        set_chksum(data)
        tcp_data_list.append(data)
        tcp_list_cv.notify_all()


def tcp_data_pop():
    with tcp_list_cv:
        while not len(tcp_data_list):
            tcp_list_cv.wait()
        return tcp_data_list.pop(0)


class TcpServerService(object):
    tcp_data_id = 0

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
            send_data = tcp_data_pop()
            # print("len of send_data", sys.getsizeof(send_data))
            print('send', int.from_bytes(send_data.id, 'little'), struct.unpack('B', send_data.type)[0],
                  int.from_bytes(send_data.param1, 'little'), int.from_bytes(send_data.param2, 'little'),
                  send_data.checksum)

            sz = sizeof(send_data)

            try:
                if client.send(bytearray(send_data)) != sz:
                    print('send length error')

                recv_data = client.recv(1024)
                if recv_data:
                    # print('recv:', recv_data)
                    pass
                else:
                    raise error('client disconnected')
            except Exception as e:
                print(e)
                client.close()
                return False
