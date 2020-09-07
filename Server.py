import socket
import pickle
import threading
import sys
import Queue
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OK = b'ok'

HOST = 'localhost'
PORT = 8080

global tasks
tasks = []
start_t = []
end_t = []
global input_queue
input_queue = Queue.Queue()

def write_to_csv(task_num):
    f = open('task_' + str(task_num) + '.csv', 'w')
    with f:
        writer = csv.writer(f)
        if tasks:
            writer.writerow(['time', 'left_pupil', 'right_pupil', 'mean', 'blink_count', 'error'])
            for t in tasks:
                writer.writerow([i for i in t.toList()])
            print('csv done.' + f.name)


def receive_data(data, append_data, fig, ax, index, error):
    print('REV--->')
    data = pickle.loads(data[3:])
    data.toString()
    if append_data:
        if error:
            data.error = 1
        tasks.append(data)
        print('append')
        print(tasks)
        index+=1
        virtualise_data(fig, ax, index, data.left_pupil, data.right_pupil, data.mean)

    else:
        print('not append')
    return fig, ax, index


def add_input(input_queue):
    while True:
        msg = sys.stdin.read(1)
        input_queue.put(msg)


def create_fig():
    fig, ax = plt.subplots()
    legend_element = [mpatches.Patch(color='blue', label='Left pupil', linestyle='-', linewidth=0.5),
                      mpatches.Patch(color='green', label='Right pupil', linestyle='-', linewidth=0.5),
                      mpatches.Patch(color='red', label='Mean', linestyle='--', linewidth=0.5)]
    ax.legend(handles=legend_element, loc='upper_left')
    ax.set_ylim(bottom=0, top=15)
    ax.set_title('Pupil Dilation')
    return fig, ax


def virtualise_data(fig, ax, i, ld, rd, m):
    # draw graph
    ax.scatter([i], [ld], color='green')
    ax.scatter([i], [rd], color='blue')
    ax.scatter([i], [m], color='red')
    fig.canvas.draw()
    print('draw')

def draw_line(fig, ax):
    if tasks:
        left_pupil = [i.left_pupil for i in tasks]
        right_pupil = [i.right_pupil for i in tasks]
        mean = [i.mean for i in tasks]

        ax.plot([i for i in range(len(left_pupil))], left_pupil, '-g')
        ax.plot([i for i in range(len(right_pupil))], right_pupil, '-b')
        ax.plot([i for i in range(len(mean))], mean, '-r')
    return fig, ax

def run(conn, task_num):
    print('create send_message thread')
    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    append_data = False
    terminate = False
    error = False

    fig, ax = create_fig()
    index = 0

    while not terminate:
        last_update = time.time()
        while True:
            if time.time() - last_update > 0.5:
                data = conn.recv(1024)
                if data:
                    if data.__contains__(b'[D]'):
                        fig, ax, index = receive_data(data, append_data, fig, ax, index, error)
                        last_update = time.time()
                        error = False

            if not input_queue.empty():
                msg = input_queue.get()[:-1]
                if msg == 's':
                    start_t.append(time.time())
                    task_num+=1
                    print(start_t)
                    append_data = True
                    error = False

                elif msg == 'e':
                    error = True
                    print('made error')

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
                print(msg)

            fig, ax = draw_line(fig, ax)
            fig.show()

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
    run(conn, task_num)
