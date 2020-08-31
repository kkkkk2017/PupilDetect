import socket
import threading
import pickle
import Queue
import time

# socket
HOST = 'localhost'
PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.connect((HOST, PORT))
msg_queue = Queue.Queue()
task_queue = Queue.Queue()

# thread for continuous receve msg and put into queue
def send_data(server, msg_queue):
    while True:
        msg = server.recv(20)
        msg_queue.put(msg.decode())

def store_data():


while True:
    send_thread = threading.Thread(target=send_data, args=(server, msg_queue, ))
    send_thread.daemon = True
    send_thread.start()

    while True:
        server.send(b'ok')
        print('send to server')
        time.sleep(0.5)


