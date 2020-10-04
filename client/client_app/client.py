class Client:
    def __init__(self):
        self.start_time = 0  # the time of starting the task
        self.socket = None
        self.left_pupil = 0  # current second pupil data
        self.right_pupil = 0
        self.blink_count = 0  # current second collected blink count
        self.time = 0  # the time when updating the data
        self.calib_blink = True
        self.left_threshold = None
        self.right_threshold = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None

    def reset(self):
        self.blink_count = 0
        self.left_pupil = 0
        self.right_pupil = 0
        self.start_time = 0

    def set_filtering(self, pre_left, pre_right):
        pre_x = [i[1][0] for i in pre_left]
        pre_y = [i[1][1] for i in pre_left]
        pre_thres = [i[0] for i in pre_left]

        self.left_x = [min(pre_x), max(pre_x)]
        self.left_y = [min(pre_y), max(pre_y)]
        self.left_threshold = [min(pre_thres) - 5, max(pre_thres) + 5]  # allow little flexible for light

        pre_x = [i[1][0] for i in pre_right]
        pre_y = [i[1][1] for i in pre_right]
        pre_thres = [i[0] for i in pre_right]

        self.right_x = [min(pre_x), max(pre_x)]
        self.right_y = [min(pre_y), max(pre_y)]
        self.right_threshold = [min(pre_thres) - 5, max(pre_thres) + 5]  # allow little flexible for light
