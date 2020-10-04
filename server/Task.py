import numpy as np

class Task:
    def __init__(self, time, left_pupil, right_pupil, blink):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.mean = 0
        self.blink_count = blink
        self.time = time
        self.error = 0

    def toList(self):
        return [self.time, self.left_pupil, self.right_pupil, self.mean, self.blink_count, self.error]

    def toString(self):
        print('left pupil: {}'.format(self.left_pupil),
              'right pupil: {}'.format(self.right_pupil),
              'blink count: {}'.format(self.blink_count),
              'error: {}'.format(self.error),
              'time: {}'.format(self.time))

