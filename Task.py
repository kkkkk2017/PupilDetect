class Task:
    def __init__(self, time, left_pupil, right_pupil, blink):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.blink_count = blink
        self.time = time

    def toString(self):
        print('\ntotal left pupil: ', {self.left_pupil},
              '\ntotal right pupil: ', {self.right_pupil},
              '\ntotal blink count: ', {self.blink_count},
              '\ntime: ', {self.time})

