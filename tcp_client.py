import socket
import signal
import sys
from time import sleep
def signal_handler(signal, frame):
    print('exit')
    client.close()  # 关闭连接
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

HOST = '192.168.40.1'
PORT = 65432
def connect_server(host, port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print('connected')
            return s
        except socket.error as e:
            print("socket error {} reconnecting".format(e))
            sleep(5)


client = connect_server(HOST, PORT)

while True:
    try :
        msg = client.recv(1024)
        print('recv:', msg)
        sleep(1)
        client.send(str.encode('hello this is client'))
        print('send hello')
    except socket.error as e:
        print('reconnect to server')
        client = connect_server(HOST, PORT)
