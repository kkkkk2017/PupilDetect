from tkinter import Frame, Button, Label, Tk
import os.path as path
import random
import time
import csv
import os

global width
global height

data_file = path.join(path.dirname(__file__), 'data.txt')
control_file = path.join(path.dirname(__file__), 'control.txt')
# root = os.environ['HOMEPATH']
# storing_path = path.join(path.dirname(__file__), 'CSV')
storing_path = os.path.expanduser("~/Desktop/CSV")
error_file = path.join(storing_path, 'error_list.csv')

tasks = ['0-back', '1-back', '2-back', '3-back']

prac_task = {
    '0-back prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'H', 'P'],
    '1-back prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'H', 'P', 'H', 'A', 'A', 'B', 'E', 'G', 'A', 'C', 'B', 'B'],
    '2-back prac': ['C', 'A', 'C', 'B', 'A', 'B', 'I', 'I', 'H', 'P', 'H', 'B', 'A', 'B', 'E', 'G', 'A', 'G', 'B', 'B'],
    '3-back prac':  ['C', 'A', 'B', 'C', 'D', 'A', 'C', 'P', 'A', 'A', 'F', 'A', 'A', 'B', 'E', 'G', 'A', 'C', 'B', 'B'],
}

instructions = {
    '0-back': '\n0-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nAnd you will need to read out loud each letter\n\n\n',
    '1-back': '1-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A A ( press \'y\' because it is the same as last one) \nB (press \'o\' because it is not the same as last one)'
              '\nYou will have only one chance for each letter',
    '2-back': '2-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 2 trials ago.'
              '\n\n If you saw the same letter 2 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A C A ( press \'y\' because it is the same as 2 trials ago) \nB (press \'o\' because it is not the same as 2 trials ago)'
              '\nYou will have only one chance for each letter',
    '3-back': '3-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 3 trials ago.'
              '\n\n If you saw the same letter 3 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A P N A( press \'y\' because it is the same as 3 trials ago) \nB (press \'o\' because it is not the same as 3 trials ago)'
              '\nYou will have only one chance for each letter',
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

        self.prepare_task() # only run 1 time
        self.create_frame()
        self.createTask()

        self.top_frame.pack()
        self.task_frame.pack()
        self.instru_frame.place(x=0, y=80)
        self.instru_frame.pack()
        self.pack()

    def check_answer(self, event):
        ans = event.char
        if ans is not None:
            self.input += 1

        if self.current_letter is None:
            return

        # 1 back
        if self.current_level == 1:
            if self.back_1 is None:
                return

            if self.input == 1:
                if (self.back_1 == self.current_letter and ans == 'y') or \
                    (self.back_1 != self.current_letter and ans == 'o'):
                    if self.if_prac:
                        self.ans_label.config(fg='black', text='Your Answer is Correct')
                else:
                    self.handle_wrong()

        elif self.current_level == 2:
            # 2 back
            if self.back_2 is None:
                return
            if self.input == 1:
                if (self.back_2 == self.current_letter and ans == 'y') or \
                    (self.back_2 != self.current_letter and ans == 'o'):
                    if self.if_prac:
                        self.ans_label.config(fg='black', text='Your Answer is Correct')
                else:
                    self.handle_wrong()

        elif self.current_level == 3:
            # 3 back
            if self.back_3 is None:
                return
            if self.input == 1:
                if (self.back_3 == self.current_letter and ans == 'y') or \
                    (self.back_3 != self.current_letter and ans == 'o'):
                    if self.if_prac:
                        self.ans_label.config(fg='black', text='Your Answer is Correct')
                else:
                    self.handle_wrong()

    def handle_wrong(self):
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
        dict = {0: '0-back',
                1: '1-back',
                2: '2-back',
                3: '3-back'}
        return dict.get(self.current_level)

    def write_control(self, command):
        with open(control_file, 'w') as f:
            f.write(command)
            f.close()

    def terminate(self):
        self.write_control('0')
        write_error(self.errors)
        self.master.destroy()

    def create_frame(self):
        self.timer_label = Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='', padx=30, fg='red', bg='white')
        self.timer_label.pack({'side': 'left'})

        self.level_label = Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Current Task: ' + self.level_to_string(), padx=30, bg='white')
        self.level_label.pack({'side': 'left'})

        self.PRAC = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='PRAC',
                            fg='blue', command=self.start_prac, bg='white')
        self.PRAC.pack({"side": "left"})

        self.START = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='START TASK',
                            fg='green', command=self.start_task, bg='white')
        self.START.pack({"side": "left"})

        self.NEXT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='NEXT TASK',
                           fg='black', command=self.reset_task, bg='white')
        self.NEXT.pack({"side": "left"})

        self.REST = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='REST',
                           fg='pink', command=self.rest, bg='white')
        self.REST.pack({"side": "left"})

        self.QUIT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='QUIT',
                           fg='red', bg='white', command=self.terminate)
        self.QUIT.pack({"side": "left"})

        self.ans_label =  Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Your Answer is ', padx=30, bg='white')
        self.ans_label.pack({'side': 'right'})

        self.master.bind("<Key>", self.check_answer)

    def prepare_task(self):
        try:
            # print('root', root)
            os.mkdir(storing_path)
        except OSError as e:
            print(e)

        self.task_labels=[]
        for i in range(10):
            label = Label(self.task_frame, font=('calibri', 100, 'bold'), text=' ', fg='white', padx=30)
            label.config(bg='white')
            label.pack({'side': 'left'})
            self.task_labels.append(label)

    def createTask(self):
        self.t = 0
        self.current_level = 0
        self.read_tasks() # get the tasks
        instruction = instructions.get(self.level_to_string())
        self.label = Label(self.instru_frame, font=('calibri', 30), text=instruction, width=300, padx=50, pady=100, fg='black')
        self.label.config(background='white', justify='center')
        self.label.pack()
        self.instru_frame.place(x=0, y=200)

    # initial task, reset the label text
    def initial_tasks(self):
        self.read_tasks()
        instruction = instructions.get(self.level_to_string())
        self.label.config(text=instruction, font=('calibri', 30), fg='black')
        self.level_label.config(text='Current Task: ' + self.level_to_string())

    def transfer_data_to_csv(self, task_num):
        # output data into csv
        data = []
        #read from data
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = line.split(',')
                data.append(d)
            f.close()

        #save to csv
        filename = path.join(storing_path, 'task_'+task_num+'.csv')
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            if data:
                writer.writerow(['time', 'left_pupil_r', 'left_pupil_x', 'left_pupil_y',
                                 'right_pupil_r', 'right_pupil_x', 'right_pupil_y', 'blink_count'])
                for t in data:
                    writer.writerow(t[:-1])
                print('[CSV] DONE!' + f.name)
            f.close()

        #clear the data file
        with open(data_file, 'w') as f:
            pass
        f.close()

    def reset_task(self):
        self.ans_label.config(text='')
        self.timer_label.config(text='')
        self.t = 0

        if len(self.not_done_level) > 0:
            get = random.randint(0, len(self.not_done_level)-1)
            self.current_level = self.not_done_level.pop(get) # get new task level
            self.initial_tasks()

        else:
            self.label.config(text='\nCOMPLETED!!\n', font=('Lucida Grande', 100, 'bold'), pady=200)


    def start_prac(self, interval=2000):
        self.if_prac = True
        if self.input == 0:
            # 1 back task, not counting the first letter
            if self.current_level == 1 and (self.t - 2) > 1 and (self.t - 2) <= len(self.current_task):
                self.handle_wrong()
            # 2 back task, not counting the second letter
            elif self.current_level == 2 and (self.t - 2) > 2 and (self.t - 2) <= len(self.current_task):
                self.handle_wrong()
            # 3 back task, not counting the third letter
            elif self.current_level == 3 and (self.t - 2) > 3 and (self.t - 2) <= len(self.current_task):
                self.handle_wrong()
        else:
            self.input = 0

        task = self.level_to_string() + ' prac'
        letters = prac_task[task]

        if self.t == 2:
            self.label.config(fg='white')
            self.task_frame.place(x=0, y=200)

        if self.t == 0:
            self.task_frame.place(x=0, y=100)
            self.task_labels[4].config(fg='white')
            self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Practice')
            self.ans_label.config(text='Your Answer is ')
            self.t += 1
            self.label.after(interval, self.start_prac)

        elif self.t == 1:
            self.label.config(text='GO', fg='black')
            self.t += 1
            self.label.after(interval, self.start_prac)

        elif self.t - 2 < len(letters):
            current = self.t - 2
            self.show_label(current, letters[current])
            self.t += 1
            self.label.after(interval, self.start_prac)

        elif self.t - 2 == len(letters):
            # hide task frame
            self.task_frame.place(x=0, y=100)
            # self.task_labels[self.t - 3].config(fg='white')
            self.task_labels[9].config(fg='white')
            self.label.config(text='End', fg='black')
            self.t += 1
            self.label.after(1000, self.start_prac)

        else:
            self.back_1 = None
            self.current_letter = None
            self.back_2 = None
            self.back_3 = None

            # show main labels
            self.t = 0


    def start_task(self, interval=2000):
        self.if_prac = False
        #check the last letter
        if self.input == 0:
            #1 back task, not counting the first letter
            if self.current_level == 1 and (self.t-2) > 1 and (self.t-2) <= len(self.current_task):
                self.handle_wrong()
            #2 back task, not counting the second letter
            elif self.current_level == 2 and (self.t-2) > 2 and (self.t-2) <= len(self.current_task):
                self.handle_wrong()
            #3 back task, not counting the third letter
            elif self.current_level == 3 and (self.t-2) > 3 and (self.t-2) <= len(self.current_task):
                self.handle_wrong()
        else:
            self.input = 0

        # t = 0 - 5
        letters = self.current_task

        if self.t == 2:
            self.label.config(fg='white')
            self.task_frame.place(x=0, y=200)

        if self.t == 0:
            self.task_frame.place(x=0, y=100)
            self.task_labels[4].config(fg='white')
            self.ans_label.config(text='')
            self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Ready')
            self.write_control('a')

            self.t += 1
            self.label.after(interval, self.start_task)

        elif self.t == 1:
            self.label.config(text='GO', fg='black')
            self.t += 1
            self.label.after(interval, self.start_task)

        elif self.t - 2 < len(letters):
            current = self.t - 2
            self.show_label(current, letters[current])
            self.t += 1
            self.label.after(interval, self.start_task)

        elif self.t - 2 == len(letters):
            # hide task frame
            self.task_frame.place(x=0, y=100)
            # self.task_labels[self.t - 3].config(fg='white')
            self.task_labels[9].config(fg='white')
            self.label.config(text='End', fg='black')
            self.t += 1
            self.label.after(1000, self.start_task)

        else:
            self.label.config(text='Take a Break', fg='black')
            self.output_csv()

            self.back_1 = None
            self.current_letter = None
            self.back_2 = None
            self.back_3 = None

            # show main labels
            self.t = 0

    def output_csv(self, rest=False):
        # stop attend data and output csv
        self.write_control('n')
        if not rest:
            self.transfer_data_to_csv(str(self.current_level))
        else:
            self.transfer_data_to_csv(str(self.current_level) + '_rest')

    def countdown(self, count):
        self.timer_label.config(text='0:' + str(count))
        if count == 0:
            self.output_csv(rest=True)
            self.label.config(text='Time\'s up')
        if count > 0:
            self.top_frame.after(1000, self.countdown, count - 1)

    def rest(self):
        self.write_control('a')
        self.label.config(text = ' ')
        self.countdown(60)

    def show_label(self, current, letter):
        label = current % 10 #[0-9]

        # unshow prev label
        if label == 0:
            self.task_labels[9].config(fg='white')
        else:
            self.task_labels[label - 1].config(fg='white')

        # show current label
        if current == 0:
            self.label.config(text=' ')

        if self.back_2:
            self.back_3 = self.back_2

        if self.back_1:
            self.back_2 = self.back_1

        if self.current_letter:
            self.back_1 = self.current_letter

        self.current_letter = letter
        self.task_labels[label].config(fg='black', text=letter)


def write_error(error_list):
    print(error_list)
    with open(error_file, 'w') as f:
        writer = csv.writer(f)
        if error_list:
            for i in error_list:
                writer.writerow([i])
            print('[CSV] DONE!' + f.name)
        f.close()


def run():
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenwidth()

    root.title('N-Back Tasks')
    root.geometry(str(width) + 'x' + str(height))
    root.resizable(0, 0)

    app = Application(root)
    app.mainloop()


if __name__ == '__main__':
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenwidth()

    root.title('N-Back Tasks')
    root.geometry(str(width) + 'x' + str(height))
    root.resizable(0, 0)

    app = Application(root)
    app.mainloop()
