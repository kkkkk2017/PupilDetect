import socket
import pickle
import threading
from imutils import face_utils
from scipy.spatial import distance as dist
import imutils
import dlib
import cv2
import numpy as np
import time
from Task import Task
import Queue

# socket
HOST = 'localhost'
PORT = 8080

# pupil detector
PUPIL_MIN = 15
EYE_AR_THRESH = 0.3  # indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
file = 'Resources/face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file)

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

class Client:
    def __init__(self, SERVER):
        self.start_time = 0 # the time of starting the task
        self.socket = SERVER
        self.left_pupil = 0 # current second pupil data
        self.right_pupil = 0
        self.blink_count = 0 # current second collected blink count
        self.time = 0 # the time when updating the data

    def send_data(self):
        data = Task(self.time, self.left_pupil, self.right_pupil, self.blink_count)
        obj = pickle.dumps(data)
        self.socket.sendall(b'[D]'+obj)
        print('send data')
        time.sleep(1)

    def extract_eye(self, frame, shape, start, end):
        (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
        roi = frame[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        return roi

    def get_keypoints(self, image, threshold):
        image = imutils.resize(image, width=250, inter=cv2.INTER_CUBIC)
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        erode = cv2.erode(image, None, iterations=3)
        dilate = cv2.dilate(erode, None, iterations=5)
        blur = cv2.medianBlur(dilate, 5)
        keypoints = blob_detector.detect(blur)
        return keypoints

    def blob_process_both_eyes(self, left_eye, right_eye, start_thres):
        left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

        if start_thres == 0:
            end_thres = 130
        else:
            end_thres = start_thres + 10
        for thres in range(start_thres - 10, end_thres):  # start_thres +- 10
            left_keypoints = self.get_keypoints(left_gray, thres)
            right_keypoints = self.get_keypoints(right_gray, thres)
            if len(left_keypoints) > 0 and len(right_keypoints) > 0:
                return thres, left_keypoints, right_keypoints
        return None, None, None


def send_data(server, msg_queue):
    while True:
        msg = server.recv(20)
        msg

if __name__ == '__main__':
    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        conn = ServerSocket.connect((HOST, PORT))
    except socket.error as err:
        print(err)

    msg_queue = Queue.Queue()

    while True:
        cap = cv2.VideoCapture(0)
        client = Client(ServerSocket)
        threshold = 0
        while True:
            _, frame = cap.read()
            frame = imutils.resize(frame, width=500, height=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left = client.extract_eye(frame, shape, lStart, lEnd)
                right = client.extract_eye(frame, shape, rStart, rEnd)

                trial_thres, le_keypoints, re_keypoints = client.blob_process_both_eyes(left, right, threshold)
                if trial_thres != None:
                    threshold = trial_thres

                print('threshold', threshold)
                if le_keypoints and re_keypoints:
                    ld, rd = le_keypoints[0].size, re_keypoints[0].size
                    if ld < PUPIL_MIN and rd < PUPIL_MIN:
                        client.time = time.time()
                        client.left_pupil = ld
                        client.right_pupil = rd
                        print(ld, rd, 'data updated!!')

                        print('create new process with new  -------> ')
                        comm_p = threading.Thread(target=client.send_data())
                        print('create new process', comm_p)
                        comm_p.start()
                        comm_p.join()

