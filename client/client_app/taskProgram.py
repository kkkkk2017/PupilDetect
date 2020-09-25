from Tkinter import *
import os.path as path
import random

global width
global height
global prac_task
global instructions

tasks = ['0-back', '1-back', '1-back-speed', '2-back', '2-back-speed']

prac_task = {
    '1-back prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O'],
    '1-back-speed prac': ['C', 'C', 'A', 'B', 'A', 'H', 'I', 'I', 'I', 'O'],
    '2-back prac': ['H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O'],
    '2-back-speed prac': ['H', 'I', 'I', 'I', 'O', 'H', 'I', 'I', 'I', 'O'],
}

instructions = {
    '0-back': '\n0-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nAnd you will nee to read out each letter\n\n\n',
    '1-back': '1-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A A ( say \'Yes\' because it is the same as last one) \nB (say \'No\' because it is not the same as last one)',
    '1-back-speed': '1-back-speed Task: You will see a sequence of letters.'
              '\nEach letter is shown for 1 second.'              
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A A ( say \'Yes\' because it is the same as last one) \nB (say \'No\' because it is not the same as last one)',
    '2-back': '2-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A C A ( say \'Yes\' because it is the same as 2 trials ago) \nB (say \'No\' because it is not the same as 2 trials ago)',
    '2-back-speed': '2-back-speed Task: You will see a sequence of letters.'
              '\nEach letter is shown for 1 second.'              
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A C A ( say \'Yes\' because it is the same as 2 trials ago) \nB (say \'No\' because it is not the same as 2 trials ago)',
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
        self.if_prac = False
        self.not_done_level = [1, 2, 3,] # exclude 2-back-speed task
        self.speed_2 = False # 2-back-speed task

        self.prepare_task() # only run 1 time
        self.create_frame()
        self.createTask()

        self.top_frame.pack()
        self.task_frame.pack()
        self.instru_frame.place(x=0, y=80)
        self.instru_frame.pack()
        self.pack()

    def read_tasks(self):
        level = self.level_to_string()
        file = path.join(path.dirname(__file__), 'tasks/'+level+'.txt')
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
        self.QUIT.pack({"side": "right"})

    def prepare_task(self):
        self.task_labels=[]
        for i in range(10):
            label = Label(self.task_frame, font=('calibri', 100, 'bold'), text=' ', fg='white', padx=33)
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

    def reset_task(self):
        self.t = 0
        # level 0 does not have prac task
        self.if_prac = True

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

    def start_task(self, interval=500):

        if self.level_to_string().__contains__('speed'):
            interval = 500  # 1000

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

        else:
            # hide task frame
            self.task_frame.place(x=0, y=100)
            # self.task_labels[self.t - 3].config(fg='white')
            self.task_labels[9].config(fg='white')
            self.label.config(text='End', fg='black')

            # show main labels
            self.t = 0
            if self.if_prac:
                self.if_prac = False

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

        self.task_labels[label].config(fg='black', text=letter)

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
    run()
