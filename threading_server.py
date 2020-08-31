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

class Server:
    def __init__(self):
        self.tasks = []
        self.start_t = []
        self.end_t = []
        self.input_queue = Queue.Queue()
        self.server_socket = None
        self.conn = None
        self.task_num = 0

    def create_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setblocking(1)
        print('start SERVER --------->')
        try:
            self.server_socket.bind((HOST, PORT))
        except socket.error as err:
            print(err)

        self.server_socket.listen(1)

    def write_to_csv(self):
        f = open('task_' + self.task_num + '.csv', 'w')
        with f:
            writer = csv.writer(f)
            if self.tasks:
                writer.writerow(['time, left_pupil, right_pupil, blink_count'])
                for t in self.tasks:
                    writer.writerow(list(t))
                print('csv done.')


    def receive_data(self, data, append_data):
        print('REV--->')
        # data = pickle.loads(data[2:])
        # data.toString()
        if append_data:
            self.tasks.append(data)
            print('append')
            print(self.tasks)
        else:
            print('not append')


    def add_input(self):
        while True:
            msg = sys.stdin.read(1)
            self.input_queue.put(msg)


    def foobar(self):
        print('create send_message thread')
        input_thread = threading.Thread(target=self.add_input, args=(self.input_queue,))
        input_thread.daemon = True
        input_thread.start()

        append_data = False

        while True:
            last_update = time.time()
            while True:
                if time.time() - last_update > 0.5:
                    data = self.conn.recv(1024)
                    if data:
                        # if data.__contains__(b'[D]'):
                        self.receive_data(data, append_data)
                    last_update = time.time()

                if not self.input_queue.empty():
                    msg = self.input_queue.get()[:-1]
                    if msg == 's':
                        self.start_t.append(time.time())
                        self.task_num+=1
                        print(self.start_t)
                        append_data = True

                    elif msg == 'e':
                        self.end_t.append(time.time())
                        print(self.end_t)
                        self.write_to_csv()
                        self.tasks = []
                        append_data = False

                    elif msg == 't':
                        self.server_socket.close()
                        print('close connection')

                    print(msg)

    def run(self):
        while True:
            self.conn, addr = self.server_socket.accept()
            print('connection established ', addr)
            self.foobar()



if __name__ == '__main__':
    server = Server()
    server.create_socket()
    server.run()