class Task:
    def __init__(self, time, left_pupil, left_pupil_x, left_pupil_y,
                 right_pupil, right_pupil_x, right_pupil_y, left_iris, left_iris_x, left_iris_y,
                 right_iris, right_iris_x, right_iris_y, blink_count):
        self.time = time
        self.left_pupil = left_pupil
        self.left_pupil_x = left_pupil_x
        self.left_pupil_y = left_pupil_y
        self.right_pupil = right_pupil
        self.right_pupil_x = right_pupil_x
        self.right_pupil_y = right_pupil_y
        self.left_iris = left_iris
        self.left_iris_x = left_iris_x
        self.left_iris_y = left_iris_y
        self.right_iris = right_iris
        self.right_iris_x = right_iris_x
        self.right_iris_y = right_iris_y
        self.blink_count = blink_count

    def to_List(self):
        return [self.time, self.left_pupil, self.left_pupil_x, self.left_pupil_y,
                self.right_pupil, self.right_pupil_x, self.right_pupil_y,
                self.left_iris, self.left_iris_x, self.left_iris_y,
                self.right_iris, self.right_iris_x, self.right_iris_y, self.blink_count]