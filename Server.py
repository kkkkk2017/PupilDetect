import socket
import numpy as np
import pickle
from _thread import *

OK = b'ok'

HOST = 'localhost'
PORT = 8080

tasks = []

def receive_data(data):
    print('REV--->', data)
    data = pickle.loads(data[2:])
    data.toString()
    tasks.append(data)
    print(tasks)


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setblocking(1)
print('start SERVER --------->')
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print(err)

s.listen(1)
while True:
    conn, addr = s.accept()
    print('connection established ', addr)

    while conn:
        print('\n')

        data = conn.recv(4096)

        if data:
            print('RECEIVE <----', data)
            if data.__contains__(b'[D]'):
                receive_data(data)
                conn.send(OK)
                continue

            elif data == b'close':
                print('Client exit')
                break


    conn.close()
    print('connection close')
