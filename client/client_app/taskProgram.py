from Tkinter import *
import os.path as path
import random
import time
import csv

global width
global height
global prac_task
global instructions

global error_list
error_list = []

tasks = ['0-back', '1-back', '1-back-speed', '2-back', '2-back-speed']

prac_task = {
    '1-back prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O'],
    '1-back-speed prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O'],
    '2-back prac': ['H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O', 'C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O',],
    '2-back-speed prac': ['H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O', 'C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O',],
}

instructions = {
    '0-back': '\n0-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nAnd you will nee to read out each letter\n\n\n',
    '1-back': '1-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A A ( press \'y\' because it is the same as last one) \nB (press \'o\' because it is not the same as last one)',
    '1-back-speed': '1-back-speed Task: You will see a sequence of letters.'
              '\nEach letter is shown for 1 second.'              
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A A ( press \'y\' because it is the same as last one) \nB (press \'o\' because it is not the same as last one)',
    '2-back': '2-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A C A ( press \'y\' because it is the same as 2 trials ago) \nB (press \'o\' because it is not the same as 2 trials ago)',
    '2-back-speed': '2-back-speed Task: You will see a sequence of letters.'
              '\nEach letter is shown for 1 second.'              
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you press \'y\', otherwise, \'o\''
              '\nFor example: A C A ( press \'y\' because it is the same as 2 trials ago) \nB (press \'o\' because it is not the same as 2 trials ago)',
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
        self.prev_letter = None
        self.pre_pre_letter = None

        self.if_prac = False
        self.not_done_level = [1, 2, 3,] # exclude 2-back-speed task
        self.speed_2 = False # 2-back-speed task
        self.errors = []
        self.error_made = 0
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
        if self.current_level < 3:
            if self.prev_letter is None:
                return

            if self.input == 1:
                if (self.prev_letter == self.current_letter and ans == 'y') or \
                    (self.prev_letter != self.current_letter and ans == 'o'):
                    self.ans_label.config(text='Your Answer is Correct')
                else:
                    self.handle_wrong()

        else:
            # 2 back
            if self.pre_pre_letter is None:
                return
            if self.input == 1:
                if (self.pre_pre_letter == self.current_letter and ans == 'y') or \
                    (self.pre_pre_letter != self.current_letter and ans == 'o'):
                    self.ans_label.config(text='Your Answer is Correct')
                else:
                    self.handle_wrong()

    def handle_wrong(self):
        self.ans_label.config(text='Your Answer is Wrong')
        self.error_made += 1
        self.error_label.config(text='| Errors: ' + str(self.error_made))
        if not self.if_prac:
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
                2: '1-back-speed',
                3: '2-back',
                4: '2-back-speed'}
        return dict.get(self.current_level)

    def terminate(self):
        # print(self.errors)
        # write(self.errors)
        store_error_list(self.errors)

        control_file = path.join(path.dirname(__file__), 'control.txt')
        with open(control_file, 'w') as f:
            f.write('0')
            f.close()

        self.master.destroy()

    def create_frame(self):
        self.level_label = Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Current Task: ' + self.level_to_string(), padx=30, bg='white')
        self.level_label.pack({'side': 'left'})

        self.START = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='START',
                            fg='green', command=self.start_task, bg='white')
        self.START.pack({"side": "left"})

        self.NEXT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='NEXT',
                           fg='black', command=self.reset_task, bg='white')
        self.NEXT.pack({"side": "left"})

        self.QUIT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='QUIT',
                           fg='red', bg='white', command=self.terminate)
        self.QUIT.pack({"side": "left"})

        self.error_label =  Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='|Errors: ' + str(self.error_made), padx=40, bg='white')
        self.error_label.pack({'side': 'right'})

        self.ans_label =  Label(self.top_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Your Answer is ', padx=30, bg='white')
        self.ans_label.pack({'side': 'right'})

        self.master.bind("<Key>", self.check_answer)

    def prepare_task(self):
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
        self.reset_error_labels()

    def reset_task(self):
        self.t = 0
        # level 0 does not have prac task
        self.if_prac = True
        self.error_made = 0

        if not self.speed_2:
            if len(self.not_done_level) > 0:
                get = random.randint(0, len(self.not_done_level)-1)
                self.current_level = self.not_done_level.pop(get) # get new task level
                self.initial_tasks()

            else:
                self.label.config(text='\nCOMPLETED!!\n', font=('Lucida Grande', 100, 'bold'), pady=200)
                self.speed_2 = True

        # a hidden task (2-back-speed)
        else:
            self.current_level = 4
            self.initial_tasks()

    def reset_error_labels(self):
        self.ans_label.config(text='Your Answer is: ')
        self.error_label.config(text='| Errors: ' + str(self.error_made))

    def start_task(self, interval=2000):
        #check the last letter
        if self.input == 0:
            #1 back task, not counting the first letter
            if self.current_level in [1,2] and (self.t-2) > 1 and (self.t-2) <= len(self.current_task):
                self.handle_wrong()
            #2 back task, not counting the second letter
            elif self.current_level in [3,4] and (self.t-2) > 2 and (self.t-2) <= len(self.current_task):
                self.handle_wrong()
        else:
            self.input = 0

        if self.level_to_string().__contains__('speed'):
            interval = 1000  # 1000

        # t = 0 - 5
        # if showing the prac task
        if self.if_prac:
            task = self.level_to_string() + ' prac'
            letters = prac_task[task]
        else:
            letters = self.current_task

        if self.t == 2:
            self.label.config(fg='white')
            self.task_frame.place(x=0, y=300)

        if self.t == 0:
            self.task_frame.place(x=0, y=100)
            self.task_labels[4].config(fg='white')
            if self.if_prac:
                self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Practice')
            else:
                self.error_made = 0
                self.reset_error_labels()
                self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Ready')

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
            if not self.if_prac:
                self.label.config(text='Take a Break', fg='black')

            self.prev_letter = None
            self.current_letter = None
            self.pre_pre_letter = None

            # show main labels
            self.t = 0
            if self.if_prac:
                self.if_prac = False

            if self.speed_2:
                global error_list
                error_list = self.errors


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

        if self.prev_letter:
            self.pre_pre_letter = self.prev_letter

        if self.current_letter:
            self.prev_letter = self.current_letter

        self.current_letter = letter
        self.task_labels[label].config(fg='black', text=letter)


def store_error_list(error_list):
    control_file = path.join(path.dirname(__file__), 'error_list.txt')
    with open(control_file, 'w') as f:
        if error_list:
            for i in error_list:
                f.write(i+'\n')
    f.close()

def write(error_list):
    f = open('../../server/data/error_list.csv', 'w')
    with f:
        writer = csv.writer(f)
        if error_list:
            writer.writerow([i for i in error_list])
            print('[CSV] DONE!' + f.name)

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
