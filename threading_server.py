import socket
import numpy as np
import pickle
import threading
import sys
import Queue
import time
import multiprocessing

OK = b'ok'

HOST = 'localhost'
PORT = 8080

global tasks
tasks = []
start_t = []
end_t = []
global input_queue
global send_queue
input_queue = multiprocessing.Queue()
send_queue = multiprocessing.Queue()

def receive_data(data, append_data):
    print('REV--->', data)
    # data = pickle.loads(data[2:])
    # data.toString()
    if append_data:
        tasks.append(data)
        print('append')
        print(tasks)
    else:
        print('not append')

def add_input(input_queue):
    while True:
        msg = sys.stdin.read(1)
        input_queue.put(msg)


def foobar(conn):
    print('create send_message thread')
    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    append_data = False

    while True:
        last_update = time.time()
        while True:
            if time.time() - last_update > 0.5:
                data = conn.recv(1024)
                if data:
                    receive_data(data, append_data)
                last_update = time.time()

            if not input_queue.empty():
                msg = input_queue.get()[:-1]
                if msg == 's':
                    start_t.append(time.time())
                    print(start_t[-1])
                    append_data = True

                elif msg == 'e':
                    end_t.append(time.time())
                    print(end_t[-1])
                    tasks = []
                    append_data = False

                print(msg)


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
    foobar(conn)

    conn.close()
    print('connection close')
