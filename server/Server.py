import socket
import pickle
import threading
import sys
import Queue
import time
import csv
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

OK = b'ok'

HOST = 'localhost'
# HOST='localhost'
PORT = 8080

global tasks
tasks = []
start_t = []
end_t = []
global input_queue
input_queue = Queue.Queue()

def receive_error_list(data):
    print('REV [ERROR-LIST] --->')
    data = pickle.loads(data[3:])
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
            writer.writerow(['time', 'left_pupil', 'right_pupil', 'mean', 'blink_count', 'error'])
            for t in tasks:
                writer.writerow([i for i in t.toList()])
            print('[CSV] DONE!' + f.name)


def receive_data(data, append_data, index):
    print('REV--->')
    data = pickle.loads(data[3:])
    data.toString()
    if append_data:
        tasks.append(data)
        print('[APPEND]')
        print(tasks)
        index+=1

    else:
        print('not append')
    return index


def add_input(input_queue):
    while True:
        msg = sys.stdin.read(1)
        input_queue.put(msg)


def virtualise_data(fig, ax, i, ld, rd, m):
    # draw graph
    ax.scatter([i], [ld], color='green')
    ax.scatter([i], [rd], color='blue')
    ax.scatter([i], [m], color='red')
    fig.canvas.draw()
    print('draw')


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
                        receive_error_list()
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
                    exit(0)

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
