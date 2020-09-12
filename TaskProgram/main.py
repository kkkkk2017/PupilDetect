from Tkinter import *
import pyglet

global width
global height
global task_dict
global source

source = pyglet.media.StaticSource(pyglet.media.load('beep.wav', streaming=False))

task_dict = {
    '0-back': ['C', 'A', 'H', 'E', 'B', 'W', 'Q', 'I', 'O', 'V'],
    '1-back': ['Q', 'C', 'F', 'W', 'W', 'M', 'M', 'R', 'Z', 'Z'],
    '1-back speed': ['B', 'B', 'B', 'H', 'A', 'M', 'N', 'V', 'V', 'J'],
    '2-back': ['K', 'A', 'A', 'K', 'R', 'A', 'R', 'C', 'B', 'M'],
    '2-back speed': ['T', 'P', 'L', 'P', 'L', 'M', 'M', 'C', 'M', 'U'],
}

instructions = {
    '0-back': '0-back Task: You will see a sequence of letters.'
              '\nEach letter is shown for 2 seconds.'
              '\nAnd you will nee to read out each letter',
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

beep = 'beep.mp3'

class Application(Frame):
    def __init__(self, master):
        self.t = 0
        self.current_level = 0

        Frame.__init__(self, master)
        self.top_frame = Frame(master, pady=10, height=30)
        self.top_frame.grid(row=1, column=1)
        self.center_frame = Frame(master)
        self.center_frame.grid(row=2, column=1)

        self.createTask()
        self.create_frame()

        self.top_frame.pack()
        self.center_frame.pack(fill=BOTH, expand=1)
        self.pack()

    def start_task(self, interval = 2000):
        self.label.config(font=('Lucida Grande', 170, 'bold'), pady=300)
        # self.player.queue(source)
        # self.player.play()
        print('\a')

        if self.level_to_string().__contains__('speed'):
            interval = 1000

        # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        #            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        #            'V', 'W', 'X', 'Y', 'Z']
        letters = task_dict.get(self.level_to_string())
        if self.t == 0:
            self.label.config(text='Ready')
            self.t += 1
            self.label.after(interval, self.start_task)
        elif self.t == 1:
            self.label.config(text='GO')
            self.t += 1
            self.label.after(interval, self.start_task)

        elif self.t-2 < len(letters):
            self.label.config(text=letters[self.t-2])
            self.t += 1
            self.label.after(interval, self.start_task)
        else:
            # self.player.pause()
            self.label.config(text='End')
            self.t = 0

    def level_to_string(self):
        dict = {0: '0-back',
                1: '1-back',
                2: '1-back speed',
                3: '2-back',
                4: '2-back speed'}
        return dict.get(self.current_level)

    def create_frame(self):
        self.QUIT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='QUIT',
                           fg='red', command=quit)
        self.QUIT.pack({"side": "right"})

        self.START = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='START',
                            fg='green', command=self.start_task)
        self.START.pack({"side": "left"})

        self.NEXT = Button(self.top_frame, pady=3, padx=30, font = ('calibri', 13, 'bold'), text='NEXT',
                           fg='black', command=self.reset_task)
        self.NEXT.pack({"side": "left"})

        self.level_label = Label(self.center_frame, font=('Lucida Grande', 13, 'bold'),
                                 text='Current Task: ' + self.level_to_string())
        self.level_label.pack()

        self.player = pyglet.media.Player()

    def createTask(self):
        self.t = 0
        self.current_level = 0
        instruction = instructions.get(self.level_to_string())
        self.label = Label(self.center_frame, font=('calibri', 30), text=instruction, width=300, padx=50, pady=300)
        self.label.config(background='white', justify='center')
        self.label.pack()

    def reset_task(self):
        self.t = 0
        if self.current_level <= 2:
            self.current_level += 1
            instruction = instructions.get(self.level_to_string())
            self.label.config(text=instruction, font=('calibri', 30))
            self.level_label.config(text='Current Task: ' + self.level_to_string())
        else:
            self.label.config(text='COMPLETED!!', font=('Lucida Grande', 100, 'bold'), pady=330)



root = Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenwidth()

root.title('N-Back Tasks')
root.geometry(str(width) + 'x' + str(height))
root.resizable(0, 0)

app = Application(root)

app.mainloop()
root.destroy()