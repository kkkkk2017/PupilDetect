import tkinter
import taskProgram
import detection
import multiprocessing as mp
from multiprocessing import Process
import os.path as path
from client import Client
from sys import platform

global client
client = Client()

def start_program():
    #172.17.253.113
    control_file = path.join(path.dirname(__file__), 'control.txt')
    with open(control_file, 'w') as f:
        f.write('1')
        f.close()

    processes = []

    server = Process(name='client', target=detection.run, args=(client,))
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
    client = detection.run_calibration(client)

def main():
    print('The current system is: ', platform)
    if platform == "win32":
        pass
    else:
        mp.set_start_method('forkserver')

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