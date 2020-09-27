import pickle
import imutils
import cv2
import numpy as np
import time
from Task import Task
from scipy.spatial import distance as dist

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.minArea = 45
detector_params.maxArea = 200


class Client:
    def __init__(self):
        self.start_time = 0  # the time of starting the task
        self.socket = None
        self.left_pupil = 0  # current second pupil data
        self.right_pupil = 0
        self.blink_count = 0  # current second collected blink count
        self.time = 0  # the time when updating the data
        self.calib_blink = True
        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)
        self.left_threshold = None
        self.right_threshold = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.error = 0

    def set_error(self, error):
        if error:
            self.error = 1
        else:
            self.error = 0

    def send_data(self, blink=False):
        if blink:
            data = Task(self.time, self.left_pupil / 3 if self.left_pupil != 0 else 0,
                        self.right_pupil / 3 if self.right_pupil != 0 else 0, self.blink_count, self.error)
            obj = pickle.dumps(data)
            self.socket.sendall(b'[D]' + obj)
        else:
            if self.left_pupil == 0 and self.right_pupil == 0:
                return
            else:
                data = Task(self.time, self.left_pupil / 3 if self.left_pupil != 0 else 0,
                            self.right_pupil / 3 if self.right_pupil != 0 else 0, blink=np.nan, error=self.error)
                obj = pickle.dumps(data)
                self.socket.sendall(b'[D]' + obj)

        time.sleep(0.5)

    def extract_eye(self, frame, shape, start, end):
        (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
        roi = frame[y:(y + h), x:(x + w)]
        roi = roi[int(h * 0.05): (h), int(w * 0.2): int(w * 0.9)]
        return roi

    def get_keypoints(self, image, threshold, side):
        if side == 0:
            min_x = self.left_x[0] if self.left_x else 0
            max_x = self.left_x[1] if self.left_x else 20
            min_y = self.left_y[0] if self.left_y else 0
            max_y = self.left_y[1] if self.left_y else 20

        if side == 1:
            min_x = self.right_x[0] if self.right_x else 0
            max_x = self.right_x[1] if self.right_x else 20
            min_y = self.right_y[0] if self.right_y else 0
            max_y = self.right_y[1] if self.right_y else 20

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = cv2.boundingRect(image)
        image = imutils.resize(image, width=int(w * 3), height=int(h * 3), inter=cv2.INTER_CUBIC)
        # image = cv2.GaussianBlur(image, (1, 1), 1)
        _, thres = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        blur = cv2.medianBlur(thres, 1)

        keypoints = self.blob_detector.detect(blur)

        if keypoints:
            if keypoints[0].pt[0] >= min_x and keypoints[0].pt[0] <= max_x:
                if keypoints[0].pt[1] >= min_y and keypoints[0].pt[1] <= max_y:
                    # image_k = cv2.drawKeypoints(image, keypoints, image, (0, 0, 255),
                    #                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # cv2.imshow('image', image_k)
                    # print(side, (x, y, w, h), keypoints[0].pt, keypoints[0].size)
                    return keypoints[0].size #diameter

        # stack = np.hstack((image, thres, blur))
        # cv2.imshow('stack', stack)
        return None

    def blob_process(self, eye, side):
        optimal = []

        if side == 0:  # 0 - left side
            start_threshold = self.left_threshold[0] if self.left_threshold is not None else 10
            end_threshold = self.left_threshold[1] if self.left_threshold is not None else 130
        if side == 1:  # 1 - right side
            start_threshold = self.right_threshold[0] if self.right_threshold is not None else 10
            end_threshold = self.right_threshold[1] if self.right_threshold is not None else 130

        for thres in range(start_threshold, end_threshold):
            area = self.get_keypoints(image=eye, threshold=thres, side=side)
            if area:
                if area < 10:
                    optimal.append(area)
                    pass
            elif area is None and len(optimal) > 0:
                break

        if len(optimal) > 0:
            if len(optimal) >= 2:
                return np.mean(optimal)
            else:
                return optimal[0]

        return None

    def blob_process_both_eyes(self, left_eye, right_eye):

        left_optimal = self.blob_process(left_eye, 0)
        right_optimal = self.blob_process(right_eye, 1)

        # # print('left', left_optimal, 'right', right_optimal)
        # if left_optimal is not None or right_optimal is not None:
        #     return left_optimal, right_optimal

        return left_optimal, right_optimal

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # vertical
        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])  # horizontal
        ear = (A + B) / (2.0 * C)
        return ear

    def show_text(self, frame, ear, time):
        cv2.putText(frame, "Blinks: {}".format(self.blink_count), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.putText(frame, 'Left Pupil: {:.5f}'.format(self.left_pupil/3), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Right Pupil: {:.5f}'.format(self.right_pupil/3), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.putText(frame, 'Time {:2f}'.format(time - self.start_time), (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    def set_filtering(self, pre_left, pre_right):
        pre_x = [i[1][0] for i in pre_left]
        pre_y = [i[1][1] for i in pre_left]
        pre_thres = [i[0] for i in pre_left]

        self.left_x = [min(pre_x), max(pre_x)]
        self.left_y = [min(pre_y), max(pre_y)]
        self.left_threshold = [min(pre_thres) - 2, max(pre_thres) + 3]  # allow little flexible for light

        pre_x = [i[1][0] for i in pre_right]
        pre_y = [i[1][1] for i in pre_right]
        pre_thres = [i[0] for i in pre_right]

        self.right_x = [min(pre_x), max(pre_x)]
        self.right_y = [min(pre_y), max(pre_y)]
        self.right_threshold = [min(pre_thres) - 2, max(pre_thres) + 3]  # allow little flexible for light
