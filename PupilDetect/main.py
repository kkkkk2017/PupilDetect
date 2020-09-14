from Tkinter import *
import client_proxyl
from multiprocessing import Process
import taskProgram

def start():
    client = Process(name='client', target=client_proxyl.run_with_server, args=(('localhost', 8080)))
    task = Process(name='task', target=taskProgram.run, args=())

    client.start()
    task.start()

    client.join()
    task.join()


if __name__ == '__main__':
    root = Tk()

    root.title('Pupil Detector')
    root.resizable(0, 0)

    calib = Button(text='CALIBRATE', fg='black', bg='light blue', width=20,
                   font=('calibri', 13, 'bold'), command=client_proxyl.run_standalone)
    calib.pack()

    start = Button(text='START', fg='black', bg='light green', width=20, font=('calibri', 13, 'bold'),
                   command=start)
    start.pack()

    quit = Button(text='QUIT', fg='black', bg='light grey', width=20, font=('calibri', 13, 'bold'),
                   command=root.destroy)
    quit.pack()

    root.mainloop()