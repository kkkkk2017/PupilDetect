
class Client:
    def __init__(self):
        self.start_time = 0  # the time of starting the task
        self.blink_count = 0  # current second collected blink count
        self.current_time = 0  # the time when updating the data
        self.calib_blink = True
        self.control = 1
        self.EYE_AR_THRESH = 0.21 # default threshold for indicate blink

        self.current_left_pupil_r = 0  # current pupil data
        self.current_right_pupil_r = 0

        self.current_left_pupil_x = 0
        self.current_left_pupil_y = 0
        self.current_right_pupil_x = 0
        self.current_right_pupil_y = 0

        self.iris_r = None

        self.approx_iris = ((165.5, 39.5), 40)
        self.approx_left_pupil = ((100, 60.5), 20)
        self.approx_right_pupil = ((100, 60.5), 20)

    def reset_data(self):
        self.blink_count = 0
        #left pupil
        self.current_left_pupil_r = 0
        self.current_left_pupil_x = 0
        self.current_left_pupil_y = 0
        #right pupil
        self.current_right_pupil_r = 0
        self.current_right_pupil_x = 0
        self.current_right_pupil_y = 0

        self.prev_left_pupil = ((self.current_left_pupil_x, self.current_left_pupil_y), self.current_left_pupil_r)
        self.prev_right_pupil = ((self.current_right_pupil_x, self.current_right_pupil_y), self.current_right_pupil_r)

    def reset_start_time(self):
        self.start_time = 0

    def get_result(self, ary):
        # print('ary', ary)
        x = sorted([i[0] for i in ary])# sort ascending
        y = sorted([i[1] for i in ary])

        x = x[1:-1]
        y = y[1:-1]

        return [(x[0], x[-1]), (y[0], y[-1])]

    def set_pupil_filtering(self, pre_left, pre_right):
        print(pre_left, pre_right)
        for result in pre_left:
            (x, y), r = result
            self.current_left_pupil_x = x
            self.current_left_pupil_y = y
            self.current_left_pupil_r = r

        for result in pre_right:
            (x, y), r = result
            self.current_right_pupil_x = x
            self.current_left_pupil_y = y
            self.current_left_pupil_r = r
