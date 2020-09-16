from Tkinter import *
import taskProgram
import client_proxyl
from multiprocessing import Process


class Main():
    def start(self):
        client = Process(name='client', target=client_proxyl.run_with_server, args=(('localhost', 8080)))
        task = Process(name='task', target=taskProgram.run, args=())

        client.start()
        task.start()

        client.join()
        task.join()

def run():
    root = Tk()

    root.title('Pupil Detector')
    root.resizable(0, 0)

    calib = Button(text='CALIBRATE', fg='blue', bg='white', width=20,
                   font=('calibri', 13, 'bold'), command=client_proxyl.run_standalone)
    calib.pack()

    start = Button(text='START', fg='dark green', bg='white', width=20, font=('calibri', 13, 'bold'),
                   command=Main.start)
    start.pack()

    quit = Button(text='QUIT', fg='red', bg='white', width=20, font=('calibri', 13, 'bold'),
                  command=root.destroy)
    quit.pack()

    root.mainloop()

if __name__ == '__main__':
    run()