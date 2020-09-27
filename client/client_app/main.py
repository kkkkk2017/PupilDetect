from Tkinter import *
import taskProgram
import client_proxyl
from multiprocessing import Process

global task
global root

def start_program():
    #172.17.253.113
    client = Process(name='client', target=client_proxyl.run_with_server, args=(('172.17.253.113', 8080)))
    task = Process(name='task', target=taskProgram.run, args=())

    client.start()
    task.start()

    client.join()
    task.join()

def export_error():
    errors = taskProgram.get_error_list()
    print(errors)

def terminate():
    exit(0)

def main():

    root = Tk()
    root.title('Pupil Detector')
    root.resizable(0, 0)

    calib = Button(text='CALIBRATE', fg='blue', bg='white', width=20,
                   font=('calibri', 13, 'bold'), command=client_proxyl.run_standalone)
    calib.pack()

    start = Button(text='START', fg='dark green', bg='white', width=20, font=('calibri', 13, 'bold'),
                   command=start_program)
    start.pack()

    export = Button(text='EXPORT', fg='grey', bg='white', width=20, font=('calibri', 13, 'bold'),
                    command=export_error)
    export.pack()

    quit = Button(text='QUIT', fg='red', bg='white', width=20, font=('calibri', 13, 'bold'),
                  command=terminate)
    quit.pack()

    root.mainloop()

if __name__ == '__main__':
    main()