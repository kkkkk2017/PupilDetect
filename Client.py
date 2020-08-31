import socket
import threading
import pickle
import Queue
import time
from Task import Task

# socket
HOST = 'localhost'
PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.connect((HOST, PORT))
msg_queue = Queue.Queue()
task_queue = Queue.Queue()

task = Task(time.time(), 4.23, 3.23, 2)

# thread for continuous receve msg and put into queue
def send_data(server, msg_queue):
    while True:
        msg = server.recv(20)
        msg_queue.put(msg.decode())

while True:
    # send_thread = threading.Thread(target=send_data, args=(server, msg_queue, ))
    # send_thread.daemon = True
    # send_thread.start()

    while True:
        data = pickle.dumps(task)
        server.send(b'[D]'+data)
        print('send to server')
        time.sleep(0.5)


