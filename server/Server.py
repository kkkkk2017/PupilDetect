import socket
import pickle
import threading
import sys
import Queue
import time
import csv

OK = b'ok'

# HOST = '172.17.253.113'
# HOST=' 172.18.20.225'
# HOST='175.45.149.94'
HOST = '108.162.250.75'
# HOST = '172.17.134.145'
PORT = 43664

global tasks
tasks = []
start_t = []
end_t = []
global input_queue
input_queue = Queue.Queue()

def receive_error_list(data):
    print('REV [ERROR-LIST] --->')
    data = pickle.loads(data[3:])
    print(data)
    f = open('data/error_list.csv', 'w')
    with f:
        writer = csv.writer(f)
        if data:
            writer.writerow([i for i in data])
            print('[CSV] DONE!' + f.name)

def write_to_csv(task_num):
    f = open('data/task_' + str(task_num) + '.csv', 'w')
    with f:
        writer = csv.writer(f)
        if tasks:
            writer.writerow(['time', 'left_pupil', 'right_pupil', 'left_iris', 'right_iris', 'blink_count'])
            for t in tasks:
                writer.writerow([i for i in t])
            print('[CSV] DONE!' + f.name)

def to_String(ary):
    attrs = ['time', 'left_pupil', 'right_pupil', 'left_iris', 'right_iris', 'blink_count']
    for i, a in zip(ary, attrs):
        print(a, i)

def receive_data(data, append_data, index):
    try:
        print('REV--->')
        try:
            # print(data)
            data = pickle.loads(data[3:])
            to_String(data)

        except pickle.PickleError as err:
            print(err)

        if append_data:
            tasks.append(data)
            print('[APPEND]')
            print(len(tasks), 'Tasks Data')
            index+=1

        else:
            print('not append')
        return index

    except socket.error as err:
        print(err)


def add_input(input_queue):
    while True:
        msg = sys.stdin.read(1)
        input_queue.put(msg)


def run(conn, task_num):
    # print('create send_message thread')
    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    append_data = False
    terminate = False

    index = 0

    while not terminate:
        last_update = time.time()
        while True:
            if time.time() - last_update > 0.5:
                data = conn.recv(1024)
                if data:
                    if data.__contains__(b'[D]'):
                        index = receive_data(data, append_data, index)
                        last_update = time.time()
                    elif data.__contains__(b'[E]'):
                        print('receive error')
                        receive_error_list(data)
                        last_update = time.time()

            if not input_queue.empty():
                msg = input_queue.get()
                # print('get', msg, type(msg))

                if msg == 's':
                    start_t.append(time.time())
                    task_num+=1
                    print(start_t)
                    append_data = True

                elif msg == 'n': # next task
                    end_t.append(time.time())
                    print(end_t)
                    write_to_csv(task_num)
                    global tasks
                    tasks = []
                    append_data = False

                elif msg == 't':
                    conn.close()
                    print('close connection')
                    terminate = True
                    break

                elif msg == 'exit':
                    break

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
    try:
        conn, addr = s.accept()
        print('connection established ', addr)
        run(conn, task_num)
    except socket.error as err:
        print(err)
    finally:
        exit(0)
