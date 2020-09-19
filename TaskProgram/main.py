from Tkinter import *
import random

global width
global height
global task_dict
global instructions

task_dict = {
    '0-back': [['C', 'A', 'H', 'E', 'B', 'W', 'Q', 'I', 'O', 'V'], ['C', 'A', 'H', 'E', 'B', 'W', 'Q', 'I', 'O', 'V']],
    '1-back prac': ['C', 'C', 'A', 'B', 'A'],
    '1-back': [['Q', 'C', 'F', 'W', 'W', 'M', 'M', 'R', 'Z', 'Z'],['t', 'r', 'i', 'a', 'l', '2']],
    '1-back speed prac': ['C', 'C', 'A', 'B', 'A'],
    '1-back speed': [['B', 'B', 'B', 'H', 'A', 'M', 'N', 'V', 'V', 'J'], ['t', 'r', 'i', 'a', 'l', '2']],
    '2-back prac': ['H', 'I', 'I', 'I', 'O'],
    '2-back': [['T', 'P', 'L', 'P', 'L', 'M', 'M', 'C', 'M', 'U'],['t', 'r', 'i', 'a', 'l', '2']],
    '2-back speed prac': ['H', 'I', 'I', 'I', 'O'],
    '2-back speed': [['K', 'A', 'A', 'K', 'R', 'A', 'R', 'C', 'B', 'M'], ['t', 'r', 'i', 'a', 'l', '2']],
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
    '1-back speed': '1-back speed Task: You will see a sequence of letters.'
              '\nEach letter is shown for 1 second.'              
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A A ( say \'Yes\' because it is the same as last one) \nB (say \'No\' because it is not the same as last one)',
    '2-back': '2-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nYou need to decide if you saw the same letter 1 trials ago.'
              '\n\n If you saw the same letter 1 trials ago, you say \'Yes\', otherwise, \'No\''
              '\nFor example: A C A ( say \'Yes\' because it is the same as 2 trials ago) \nB (say \'No\' because it is not the same as 2 trials ago)',
    '2-back speed': '2-back speed Task: You will see a sequence of letters.'
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
        self.center_frame = Frame(master)
        self.center_frame.grid(row=1, column=1, rowspan=1)
        self.task_frame = Frame(master)

        self.prepare_task() # only run 1 time
        self.create_frame()
        self.createTask()

        self.top_frame.pack()
        self.task_frame.pack()
        self.center_frame.place(x=0, y=80)
        self.center_frame.pack()
        self.pack()
        self.if_prac = False
        self.not_done_level = [1, 2, 3,] # exclude 2-back speed task
        self.speed_2 = False # 2-back speed task
        self.trial = 0 # the number of task trial

        # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        #            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        #            'V', 'W', 'X', 'Y', 'Z']

    def start_task(self, interval = 2000):

        if self.level_to_string().__contains__('speed'):
            interval = 1000

        # t = 0 - 5
        # if showing the prac task
        if self.if_prac:
            self.label.config(fg='white')
            task = self.level_to_string() + ' prac'
            letters = task_dict.get(task)
            if self.t == 0:
                self.task_frame.place(x=0, y=300)

            if self.t > 0:
                self.task_labels[self.t - 1].config(fg='white')
            self.task_labels[self.t].config(text=letters[self.t], fg='black')

            if self.t == 4:
                self.if_prac = False
                self.t = 0
            else:
                self.t += 1

            self.label.after(interval, self.start_task)

        else:
            if self.t == 2:
                self.label.config(fg='white')
                self.task_frame.place(x=0, y=300)

            letters = task_dict.get(self.level_to_string())[self.trial]

            if self.t == 0:
                self.task_frame.place(x=0, y=100)
                self.task_labels[4].config(fg='white')
                self.label.config(font=('Lucida Grande', 100, 'bold'), pady=250, fg='black', text='Ready')
                self.t += 1
                self.label.after(interval, self.start_task)

            elif self.t == 1:
                self.label.config(text='GO', fg='black')
                self.t += 1
                self.label.after(interval, self.start_task)

            elif self.t-2 < len(letters):
                current = self.t-2
                self.show_label(current, letters[current])
                self.t += 1
                self.label.after(interval, self.start_task)

            else:
                # hide task frame
                self.task_frame.place(x=0, y=120)
                self.task_labels[self.t - 3].config(fg='white')
                # show main labels
                self.label.config(text='End', fg='black')
                self.t = 0
                self.if_prac = True
                self.trial = 1


    def show_label(self, current, letter):
        if current > 0:
            # unshow prev label
            self.task_labels[current - 1].config(fg='white')

        #show current label
        self.label.config(text='')
        self.task_labels[current].config(fg='black', text=letter)

    def level_to_string(self):
        dict = {0: '0-back',
                1: '1-back',
                2: '1-back speed',
                3: '2-back',
                4: '2-back speed'}
        return dict.get(self.current_level)

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
                           fg='red', bg='white', command=self.master.destroy)
        self.QUIT.pack({"side": "right"})

    #prepare 0-back task
    def prepare_task(self):
        self.task_labels=[]
        letters = task_dict.get('0-back')[0]
        for i in range(len(letters)):
            label = Label(self.task_frame, font=('calibri', 100, 'bold'), text=letters[i], fg='white', padx=33)
            label.config(bg='white')
            label.pack({'side': 'left'})
            self.task_labels.append(label)

    def createTask(self):
        self.t = 0
        self.current_level = 0
        self.trial = 0
        instruction = instructions.get(self.level_to_string())
        self.label = Label(self.center_frame, font=('calibri', 30), text=instruction, width=300, padx=50, pady=100, fg='black')
        self.label.config(background='white', justify='center')
        self.label.pack()
        self.center_frame.place(x=0, y=200)

    # initial task, reset the label text
    def initial_tasks(self):
        instruction = instructions.get(self.level_to_string())
        self.label.config(text=instruction, font=('calibri', 30), fg='black')
        self.level_label.config(text='Current Task: ' + self.level_to_string())

        letters = task_dict.get(self.level_to_string())[self.trial]
        for i in range(len(letters)):
            self.task_labels[i].config(text=letters[i])


    def reset_task(self):
        self.t = 0
        self.trial = 0
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

        # a hidden task (2-back speed)
        else:
            self.current_level = 4
            self.initial_tasks()

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
