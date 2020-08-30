import tkinter as tk

def handle_click(event):
    print('The button was clicked')


if __name__ == 'main':
    window = tk.Tk()
    label = tk.Label(text='Server Window')

    window.rowconfigure(0, minsize=70, weight=1)
    window.columnconfigure([0, 1, 2], minsize=70, weight=1)

    btn_start = tk.Button(master=window, text='Start Task')
    btn_start.grid(row=0, column=0)
    btn_start.bind('<Start Button>', handle_click)

    btn_end=tk.Button(master=window, text='End Task')
    btn_end.grid(row=0, column=1)
    btn_end.bind('<End Button>', handle_click)

    btn_next=tk.Button(master=window, text='Next Task')
    btn_next.grid(row=0, column=2)
    btn_next.bind('<Next Button>', handle_click)

    window.mainloop()