import pickle
import imutils
import cv2
import numpy as np
import time
from Task import Task
from scipy.spatial import distance as dist

class Client:
    def __init__(self, SERVER=None):
        self.start_time = 0 # the time of starting the task
        self.socket = SERVER
        self.left_pupil = 0 # current second pupil data
        self.right_pupil = 0
        self.blink_count = 0 # current second collected blink count
        self.time = 0 # the time when updating the data

    def send_data(self, blink=False):
        if blink:
            data = Task(self.time, self.left_pupil, self.right_pupil, self.blink_count)
            obj = pickle.dumps(data)
            self.socket.sendall(b'[D]'+obj)
            print('send data')
        else:
            data = Task(self.time, self.left_pupil, self.right_pupil, 0)
            obj = pickle.dumps(data)
            self.socket.sendall(b'[D]' + obj)
            print('send data')

        time.sleep(1)

    def extract_eye(self, frame, shape, start, end):
        (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
        roi = frame[y:(y + h), x-1:(x + w)-1]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        return roi

    def get_keypoints(self, image, detector, threshold):
        image = imutils.resize(image, width=50, inter=cv2.INTER_CUBIC)
        image = cv2.GaussianBlur(image, (3, 3), 1)
        _, thres = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        blur = cv2.medianBlur(thres, 1)

        keypoints = detector.detect(blur)

        if keypoints:
            image_k = cv2.drawKeypoints(image, keypoints, image, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('image', image_k)
            return keypoints[0].size

        stack = np.hstack((image, thres, blur))
        cv2.imshow('stack', stack)
        return None

    def blob_process(self, eye, detector, start_thres):
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        optimal = []

        for thres in range(0, 130):  # 0 - 255
            area = self.get_keypoints(gray, detector, thres)
            if area:
                if area < 10:
                    optimal.append((thres, area))
                    pass
            if area is None and len(optimal) > 0:
                break

        if len(optimal) > 0:
            optimal = sorted(optimal, key=lambda x: x[1], reverse=True)
            return optimal[0]

        return None

    def blob_process_both_eyes(self, left_eye, right_eye, thres_left, thres_right):
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = 15
        detector_params.maxArea = 60
        blob_detector = cv2.SimpleBlobDetector_create(detector_params)

        left_optimal = self.blob_process(left_eye, blob_detector, thres_left)
        right_optimal = self.blob_process(right_eye, blob_detector, thres_right)

        # print('left', left_optimal, 'right', right_optimal)
        if left_optimal is not None and right_optimal is not None:
            return left_optimal, right_optimal

        return None, None

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
        cv2.putText(frame, 'Left Pupil: {:.3f}'.format(self.left_pupil), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Right Pupil: {:.3f}'.format(self.right_pupil), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Time {:2f}'.format(time - self.start_time), (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


