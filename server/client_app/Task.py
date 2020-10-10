class Task:
    def __init__(self, time, left_pupil, right_pupil, left_iris, right_iris, blink):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.left_iris = left_iris
        self.right_iris = right_iris
        self.blink_count = blink
        self.time = time

    def toList(self):
        return [self.time, self.left_pupil, self.right_pupil, self.left_iris, self.right_iris, self.blink_count,]

    def toString(self):
        print('left pupil: {}'.format(self.left_pupil),
              'right pupil: {}'.format(self.right_pupil),
              'left iris: {}'.format(self.left_iris),
              'right iris:{}'.format(self.right_iris),
              'blink count: {}'.format(self.blink_count),
              'time: {}'.format(self.time))

