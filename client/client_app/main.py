import tkinter
import taskProgram
import client_proxyl
from multiprocessing import Process
import os.path as path
from client import Client

global client
client = Client()

def start_program():
    #172.17.253.113
    control_file = path.join(path.dirname(__file__), 'control.txt')
    with open(control_file, 'w') as f:
        f.write('1')
        f.close()

    processes = []

    server = Process(name='client', target=client_proxyl.run, args=(client, ))
    task = Process(name='task', target=taskProgram.run, args=())

    server.start()
    processes.append(server)
    task.start()
    processes.append(task)

    for p in processes:
        p.join()

def terminate():
    exit(0)

def run_calib():
    global client
    client = client_proxyl.run_standalone(client)

def main():
    root = tkinter.Tk()
    root.title('Pupil Detector')
    # root.resizable(False, False)

    calib = tkinter.Button(text='CALIBRATE', fg='blue', bg='white', width=20,
                   font=('calibri', 13, 'bold'), command=run_calib)
    calib.pack()

    start = tkinter.Button(text='START', fg='dark green', bg='white', width=20, font=('calibri', 13, 'bold'),
                   command=start_program)
    start.pack()

    # export = Button(text='EXPORT', fg='grey', bg='white', width=20, font=('calibri', 13, 'bold'),
    #                 command=export_error)
    # export.pack()

    quit = tkinter.Button(text='QUIT', fg='red', bg='white', width=20, font=('calibri', 13, 'bold'),
                  command=terminate)
    quit.pack()

    root.mainloop()

if __name__ == '__main__':
    main()