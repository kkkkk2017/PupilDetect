class Task:
    def __init__(self, time, left_pupil_r, left_pupil_x, left_pupil_y,
                 right_pupil_r, right_pupil_x, right_pupil_y, blink_count):
        self.time = time
        self.left_pupil_r = left_pupil_r
        self.left_pupil_x = left_pupil_x
        self.left_pupil_y = left_pupil_y
        self.right_pupil_r = right_pupil_r
        self.right_pupil_x = right_pupil_x
        self.right_pupil_y = right_pupil_y
        self.blink_count = blink_count

    def to_List(self):
        return [self.time, self.left_pupil_r, self.left_pupil_x, self.left_pupil_y,
                self.right_pupil_r, self.right_pupil_x, self.right_pupil_y, self.blink_count]