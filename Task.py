import numpy

class Task:
    def __init__(self, time, left_pupil, right_pupil, blink):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.mean = numpy.mean(left_pupil, right_pupil)
        self.blink_count = blink
        self.time = time

    def toList(self):
        return [self.time, self.left_pupil, self.right_pupil, self.mean, self.blink_count]

    def toString(self):
        print('\nleft pupil: ', {self.left_pupil},
              '\nright pupil: ', {self.right_pupil},
              '\nblink count: ', {self.blink_count},
              '\ntime: ', {self.time})

