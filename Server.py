import socket
import numpy as np
import pickle
import threading
import sys
import Queue
import time
import csv

OK = b'ok'

HOST = 'localhost'
PORT = 8080

global tasks
tasks = []
start_t = []
end_t = []
global input_queue
input_queue = Queue.Queue()

def write_to_csv():
    f = open('task_' + str(task_num) + '.csv', 'w')
    with f:
        writer = csv.writer(f)
        if tasks:
            writer.writerow(['time', 'left_pupil', 'right_pupil', 'mean', 'blink_count'])
            for t in tasks:
                writer.writerow([i for i in t.toList()])
            print('csv done.')


def receive_data(data, append_data):
    print('REV--->')
    data = pickle.loads(data[3:])
    data.toString()
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


def foobar(conn, task_num):
    print('create send_message thread')
    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    append_data = False
    terminate = False
    while not terminate:
        last_update = time.time()
        while True:
            if time.time() - last_update > 0.5:
                data = conn.recv(1024)
                if data:
                    if data.__contains__(b'[D]'):
                        receive_data(data, append_data)
                last_update = time.time()

            if not input_queue.empty():
                msg = input_queue.get()[:-1]
                if msg == 's':
                    start_t.append(time.time())
                    task_num+=1
                    print(start_t)
                    append_data = True

                elif msg == 'e':
                    end_t.append(time.time())
                    print(end_t)
                    write_to_csv()
                    global tasks
                    tasks = []
                    append_data = False

                elif msg == 't':
                    conn.close()
                    print('close connection')
                    terminate = True
                    break
                print(msg)

        # if data:
        #     print('RECEIVE <----', data)
        #     if data.__contains__(b'[D]'):
        #         receive_data(data)
        #         conn.send(OK)
        #         continue
        #
        #     elif data == b'close':
        #         print('Client exit')
        #         break


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setblocking(1)
print('start SERVER --------->')
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print(err)

s.listen(1)
task_num = 0

while True:
    conn, addr = s.accept()
    print('connection established ', addr)
    foobar(conn, task_num)
