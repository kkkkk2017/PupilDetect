import numpy as np

class Client:
    def __init__(self):
        self.start_time = 0  # the time of starting the task
        self.blink_count = 0  # current second collected blink count
        self.current_time = 0  # the time when updating the data
        self.calib_blink = True
        self.left_threshold = None
        self.right_threshold = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.left_iris = None #[(min_x, max_x), (min_y, max_y), (min_size, max_size)]
        self.right_iris = None
        self.control = 1

        self.current_left_pupil = 0  # current pupil data
        self.current_right_pupil = 0
        self.current_left_iris = 0
        self.current_right_iris = 0

        self.current_left_pupil_x = 0
        self.current_left_pupil_y = 0
        self.current_right_pupil_x = 0
        self.current_right_pupil_y = 0
        self.current_left_iris_x = 0
        self.current_left_iris_y = 0
        self.current_right_iris_x = 0
        self.current_right_iris_y = 0

    def reset_data(self):
        self.blink_count = 0
        #left pupil
        self.current_left_pupil = 0
        self.current_left_pupil_x = 0
        self.current_left_pupil_y = 0
        #right pupil
        self.current_right_pupil = 0
        self.current_right_pupil_x = 0
        self.current_right_pupil_y = 0
        #left iris
        self.current_left_iris = 0
        self.current_left_iris_x = 0
        self.current_left_iris_y = 0
        #right iris
        self.current_right_iris = 0
        self.current_right_iris_x = 0
        self.current_right_iris_y = 0

    def reset_start_time(self):
        self.start_time = 0

    def get_result(self, ary):
        x = sorted([i[0] for i in ary])# sort ascending
        y = sorted([i[1] for i in ary])
        size = sorted([i[2] for i in ary])

        x = x[1:-1]
        y = y[1:-1]
        size = size[1:-1] #remove the smallest and biggest

        print('x', x)
        print('y', y)
        print('size', size)

        return [(x[0], x[-1]), (y[0], y[-1]), (size[0], size[-1])]

    def set_filtering(self, pre_left, pre_right, left_threshold, right_threshold):
        #input: [ (iris(x, y, size), pupil(x, y, size)), .... ]
        left_iris = [i[0] for i in pre_left]
        self.left_iris = self.get_result(left_iris)

        left_pupil = [i[1] for i in pre_left]
        left_pupil = self.get_result(left_pupil)
        self.left_x = left_pupil[0]
        self.left_y = left_pupil[1]

        self.left_threshold = [min(left_threshold) - 5, max(left_threshold) + 5]  # allow little flexible for light

        right_iris = [i[0] for i in pre_right]
        self.right_iris = self.get_result(right_iris)

        right_pupil = [i[1] for i in pre_right]
        right_pupil = self.get_result(right_pupil)
        self.right_x = right_pupil[0]
        self.right_y = right_pupil[1]
        self.right_threshold = [min(right_threshold) - 5, max(right_threshold) + 5]  # allow little flexible for light

        # print('filter set [left]', self.left_iris, self.left_x, self.left_y, self.left_threshold)
        # print('filter set [right]', self.right_iris, self.right_x, self.right_y, self.right_threshold)
