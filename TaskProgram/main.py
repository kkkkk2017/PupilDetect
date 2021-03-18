from tkinter import Frame, Button, Label, Tk, Canvas
import os.path as path
import random
import time
import cv2
import imutils

global width
global height


tasks = ['video', '1-back', '2-back', '3-back']

prac_task = {
    '1-back prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'H', 'P', 'H', 'A', 'A', 'B', 'E', 'G', 'A', 'C', 'B', 'B'],
    '2-back prac': ['C', 'A', 'C', 'B', 'A', 'B', 'I', 'I', 'H', 'P', 'H', 'B', 'A', 'B', 'E', 'G', 'A', 'G', 'B', 'B'],
    '3-back prac':  ['C', 'A', 'B', 'C', 'D', 'A', 'C', 'P', 'A', 'A', 'F', 'A', 'A', 'B', 'E', 'G', 'A', 'C', 'B', 'B'],
}

instructions = {
    '1-back': '1-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\'.'
              '\nFor example: A A ( press \'y\' because it is the same as last one)',
    '2-back': '2-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 2 trials ago.'
              '\n\n If you saw the same letter 2 trials ago, you press \'y\'.'
              '\nFor example: A C A ( press \'y\' because it is the same as 2 trials ago)',
    '3-back': '3-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 3 trials ago.'
              '\n\n If you saw the same letter 3 trials ago, you press \'y\'.'
              '\nFor example: A P N A( press \'y\' because it is the same as 3 trials ago)',
}

class Application(Frame):
    def __init__(self, master):
        self.t = 0
        self.current_level = 0

        Frame.__init__(self, master)
        self.master.config(bg='white')
        self.top_frame = Frame(master, pady=10, height=40)
        self.top_frame.grid(row=0, column=1)
        self.top_frame.config(bg='white')
        self.instru_frame = Frame(master)
        self.instru_frame.config(background='white')
        self.instru_frame.grid(row=1, column=1, rowspan=1)
        self.task_frame = Frame(master)

        self.current_task = []
        self.current_letter = None
        self.back_1 = None
        self.back_2 = None
        self.back_3 = None

        self.if_prac = False
        self.not_done_level = [1, 2, 3]
        self.errors = []
        self.input = 0

        self.level_label = Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Current Task: ' + self.level_to_string(), padx=30, bg='white')
        self.level_label.pack({'side': 'left'})
        self.NEXT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='NEXT TASK',
                           fg='black', command=self.reset_task, bg='white')
        self.NEXT.pack({"side": "left"})

        self.createTask()

        self.top_frame.pack()
        self.task_frame.pack()
        self.instru_frame.place(x=0, y=80)
        self.instru_frame.pack()
        self.pack()

        self.vid = None
        self.count = 0

    def play_video(self):
        video_dir = path.join(path.dirname(__file__), 'tasks', 'relax_video.mp4')
        self.vid = cv2.VideoCapture(video_dir)
        self.canvas = Canvas(self.task_frame)
        self.canvas.pack()

        while (self.vid.isOpened()):
            ok, frame = self.vid.read()
            if ok == True:
                frame = imutils.resize(frame, width=self.master.winfo_width(), height=self.winfo_height(),
                                       inter=cv2.INTER_CUBIC)
                cv2.imshow('video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        self.finish_video()


    def finish_video(self):
        self.vid.release()
        cv2.destroyAllWindows()
        self.canvas.destroy()
        self.video_btn.destroy()
        self.label.config(text='End\n\nYou can press the\'nextTask\' button to go to the next task', fg='black')


    def check_answer(self, event):
        ans = event.char
        if ans is not None:
            self.input += 1

        if self.current_letter is None or getattr(self, 'back_' + str(self.current_level)) is None:
            return

        if self.input == 1:
            correct = getattr(self, 'back_' + str(self.current_level)) == self.current_letter and ans == 'y'
            # print('check for input == 0', correct, self.current_letter, ans)

            if correct:
                if self.if_prac:
                    self.ans_label.config(fg='black', text='Your Answer is Correct')
            else:
                if self.if_prac:
                    self.ans_label.config(fg='black', text='Your Answer is Wrong')
                else:
                    self.errors.append(str(time.time()))


    def read_tasks(self):
        level = self.level_to_string()
        file = path.join(path.dirname(__file__), 'tasks', level+'.txt')
        self.current_task = []
        f = open(file, 'r')
        lines = f.readlines()
        for l in lines:
            self.current_task.append(l[0])
        f.close()

    def level_to_string(self):
        dict = {0: 'video',
                1: '1-back',
                2: '2-back',
                3: '3-back'}
        return dict.get(self.current_level)

    def terminate(self):
        self.master.destroy()

    def create_frame(self):
        self.timer_label = Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='', padx=30, fg='red', bg='white')
        self.timer_label.pack({'side': 'left'})

        self.PRAC = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='PRAC',
                            fg='blue', command=self.start_prac, bg='white')
        self.PRAC.pack({"side": "left"})

        self.START = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='START TASK',
                            fg='green', command=self.start_task, bg='white')
        self.START.pack({"side": "left"})

        self.REST = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='REST',
                           fg='purple', command=self.rest, bg='white')
        self.REST.pack({"side": "left"})

        self.QUIT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='QUIT',
                           fg='red', bg='white', command=self.terminate)
        self.QUIT.pack({"side": "left"})

        self.ans_label =  Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Your Answer is ', padx=30, bg='white')
        self.ans_label.pack({'side': 'right'})

        self.master.bind("<Key>", self.check_answer)


    def createTask(self):
        self.t = 0
        self.current_level = 0
        self.label = Label(self.instru_frame, font=('calibri', 30), text='', width=300, padx=50, pady=100, fg='black')
        self.label.config(background='white', justify='center')
        self.label.pack()
        self.instru_frame.place(x=0, y=200)

        self.video_btn = Button(self.instru_frame, pady=3, padx=30, font=('calibri', 15, 'bold'), text='Play Video',
                                fg='black', command=self.play_video, bg='white')
        self.video_btn.config(justify='center')
        self.video_btn.pack()
        # self.video_btn.place(x=600, y=100)

    # initial task, reset the label text
    def initial_tasks(self):
        self.read_tasks()
        instruction = instructions.get(self.level_to_string())
        self.label.config(text=instruction, font=('calibri', 30), fg='black')
        self.level_label.config(text='Current Task: ' + self.level_to_string())

    def reset_task(self):
        if self.current_level == 0:
            self.create_frame()
            if self.video_btn:
                self.video_btn.destroy()

        self.ans_label.config(text='')
        self.timer_label.config(text='')
        self.t = 0

        if len(self.not_done_level) > 0:
            get = random.randint(0, len(self.not_done_level)-1)
            self.current_level = self.not_done_level.pop(get) # get new task level
            self.initial_tasks()

        else:
            self.label.config(text='\nCOMPLETED!!\n', font=('Lucida Grande', 100, 'bold'), pady=200)

    #check answer for when no input
    def check_input(self):
        if (self.current_level == 1 and (self.t - 2) <= 1) \
            or (self.current_level == 2 and (self.t - 2) <= 2) \
               or (self.current_level == 3 and (self.t - 2) <= 3):
            # print('skip')
            return

        if self.input == 0:
            correct = getattr(self, 'back_' + str(self.current_level)) != self.current_letter
            # print('check for input == 0', correct, self.current_letter)
            if correct:
                if self.if_prac:
                    self.ans_label.config(fg='black', text='Your Answer is Correct')
            else:
                if self.if_prac:
                    self.ans_label.config(fg='black', text='Your Answer is Wrong')
                else:
                    self.errors.append(str(time.time()))

    def task_end(self):
        if not self.if_prac:
            self.check_input() #check the last input
            self.label.config(text='Take a Break')

        self.back_1 = None
        self.current_letter = None
        self.back_2 = None
        self.back_3 = None
        self.t = 0

    # don't combine those functions, practice can be attended multiple times.
    def start_task(self):
        self.if_prac = False
        self.run()

    def start_prac(self):
        self.if_prac = True
        self.run()

    def run(self, interval=500):
        if self.current_level != 0:
            self.count += 1
            # self.if_prac = False
            #check the last letter
            # t = 0 - 5
            if self.if_prac:
                task = self.level_to_string() + ' prac'
                letters = prac_task[task]
            else:
                letters = self.current_task

            if self.t == 0:
                # self.task_frame.place(x=0, y=100)
                if self.if_prac:
                    self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Practice')
                    self.ans_label.config(text='Your Answer is ')
                else:
                    self.ans_label.config(text='')
                    self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Ready')

                self.t += 1
                self.label.after(interval, self.run)

            elif self.t == 1:
                self.label.config(text='GO')
                self.t += 1
                self.label.after(interval, self.run)
                self.count = 0

            elif self.t - 2 < len(letters):
                if self.count == 1:
                    current = self.t - 2
                    self.input = 0
                    self.show_label(letters[current])
                    self.t += 1
                else:
                    if self.count == 3:
                        self.label.config(text='')
                    elif self.count == 4:
                        self.check_input()
                        self.count = 0

                self.label.after(interval, self.run)

            elif self.t - 2 == len(letters):
                self.label.config(text='End')
                self.t += 1
                self.label.after(1000, self.run)
            else:
                self.task_end()


    def countdown(self, count):
        self.timer_label.config(text='0:' + str(count))
        if count == 0:
            self.label.config(text='Time\'s up')
        if count > 0:
            self.top_frame.after(1000, self.countdown, count - 1)

    def rest(self):
        self.label.config(text = ' ')
        self.countdown(30)

    def show_label(self, letter):
        if self.back_2:
            self.back_3 = self.back_2

        if self.back_1:
            self.back_2 = self.back_1

        if self.current_letter:
            self.back_1 = self.current_letter

        self.current_letter = letter
        self.label.config(text=letter)

# def run():
#     root = Tk()
#     width = root.winfo_screenwidth()
#     height = root.winfo_screenwidth()
#
#     root.title('N-Back Tasks')
#     root.geometry(str(width) + 'x' + str(height))
#     root.resizable(0, 0)
#
#     app = Application(root)
#     app.mainloop()


def main():
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenwidth()

    root.title('N-Back Tasks')
    root.geometry(str(width) + 'x' + str(height))
    root.resizable(0, 0)

    app = Application(root)
    app.mainloop()
