from collections import Counter

class Client:
    def __init__(self):
        self.start_time = 0  # the time of starting the task
        self.blink_count = 0  # current second collected blink count
        self.current_time = 0  # the time when updating the data
        self.calib_blink = True
        self.left_pupil_threshold = None
        self.right_pupil_threshold = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None

        self.left_iris_size = None
        self.right_iris_size = None
        # self.left_iris_threshold = None
        # self.right_iris_threshold = None
        # self.left_iris_coordinate = None #[(min_x, max_x), (min_y, max_y),]
        # self.right_iris_coordinate = None

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
        # print('ary', ary)
        x = sorted([i[0] for i in ary])# sort ascending
        y = sorted([i[1] for i in ary])

        x = x[1:-1]
        y = y[1:-1]

        return [(x[0], x[-1]), (y[0], y[-1])]

    def set_iris_fitering(self, iris_ary, iris_threshold, if_left):
        # print('set iris filtering', iris_ary)
        size = Counter([int(i[2]) for i in iris_ary]).most_common(1)
        size = size[0][0]
        iris_threshold = sorted(iris_threshold)

        if if_left:
            self.left_iris_size = size
            self.left_iris_threshold = (min(iris_threshold)-3, max(iris_threshold)+3)
            # self.left_iris_coordinate = self.get_result(iris_ary)

        else:
            self.right_iris_size = size
            self.right_iris_threshold = (min(iris_threshold)-3, max(iris_threshold)+3)
            # self.right_iris_coordinate = self.get_result(iris_ary)

    def set_pupil_filtering(self, pre_left, pre_right, left_threshold, right_threshold):
        #input: [ (iris(x, y, size), pupil(x, y, size)), .... ]
        # left_iris = [i[0] for i in pre_left]
        # self.left_iris_coordinate = self.get_result(left_iris)

        # left_pupil = [i[0] for i in pre_left]
        left_pupil = self.get_result(pre_left)
        self.left_x = left_pupil[0]
        self.left_y = left_pupil[1]

        self.left_pupil_threshold = [min(left_threshold) - 5, max(left_threshold) + 5]  # allow little flexible for light

        # right_iris = [i[0] for i in pre_right]
        # self.right_iris_coordinate = self.get_result(right_iris)

        right_pupil = self.get_result(pre_right)
        self.right_x = right_pupil[0]
        self.right_y = right_pupil[1]
        self.right_pupil_threshold = [min(right_threshold) - 5, max(right_threshold) + 5 ]  # allow little flexible for light

        # print('filter set [left]', self.left_iris_coordinate, self.left_x, self.left_y, self.left_threshold)
        # print('filter set [right]', self.right_iris_coordinate, self.right_x, self.right_y, self.right_threshold)
