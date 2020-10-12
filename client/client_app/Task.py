class Task:
    def __init__(self, time, left_pupil, right_pupil, left_iris, right_iris, error):
        self.time = time
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.left_iris = left_iris
        self.right_iris = right_iris
        self.error = error

    def to_List(self):
        return [self.time, self.left_pupil, self.right_pupil, self.left_iris, self.right_iris, self.error]