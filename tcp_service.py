import socket
import threading

from window_info import send_window_info, set_terminal_size

tcp_data_list = []
tcp_list_cv = threading.Condition()
client_connected = False
from defs import *


def tcp_data_append(data):
    with tcp_list_cv:
        TcpServerService.tcp_data_id += 1
        set_id(data, TcpServerService.tcp_data_id)
        set_chksum(data)
        if len(tcp_data_list) > 100:
            tcp_data_list.clear()
        tcp_data_list.append(data)
        tcp_list_cv.notify_all()


def tcp_data_pop():
    with tcp_list_cv:
        while not len(tcp_data_list):
            tcp_list_cv.wait()
        return tcp_data_list.pop(0)


class TcpServerService(object):
    tcp_data_id = 0
    stop_send = 0
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_DONTROUTE, 1)
            self.sock.bind((self.host, self.port))
            lst = threading.Thread(target=self.thread_listen, args=())
            lst.daemon = True
            lst.start()
            return 0
        except:
            print('start tcp server error')
            return 1

    def thread_listen(self):
        global client_connected
        while True:
            self.sock.listen(5)
            client, addr = self.sock.accept()
            print('accepted: %s' % (addr,))
            client.settimeout(60)
            client_connected = True
            tcp_data_list.clear()
            send_window_info() # tell emulator its position and size

            recv_data = client.recv(32)
            if recv_data:
                # char_array = ctypes.c_char * len(recv_data)
                width = get_param1(recv_data)
                height = get_param2(recv_data)
                set_terminal_size(width, height)

            snd = threading.Thread(target=self.thread_send, args=(client, addr))
            snd.daemon = True
            snd.start()

    def thread_send(self, client, address):
        while True:
            send_data = tcp_data_pop()
            # print("len of send_data", sys.getsizeof(send_data))
            # print(int.from_bytes(send_data.id, 'little'), struct.unpack('B', send_data.type)[0],
            #        int.from_bytes(send_data.param1, 'little'), int.from_bytes(send_data.param2, 'little'),
            #        send_data.checksum)


            try:
                if TcpServerService.stop_send:
                    continue
                sz = sizeof(send_data)
                sent = client.send(bytearray(send_data))
                #res = ' '.join(format(x, '02x') for x in bytearray(send_data))
                #print('send', str(res))
                #print('sent=',sent,'size=',sz)
                if sent != sz:
                    print('send length error')

                # recv_data = client.recv(1024)
                # if recv_data:
                #     print('recv:', recv_data)
                #     pass
                # else:
                #     raise error('client disconnected')
            except Exception as e:
                print(e)
                client.close()
                global client_connected
                client_connected = False
                return False
