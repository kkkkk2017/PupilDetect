from imutils import face_utils
from scipy.spatial import distance as dist
import imutils
import dlib
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

PUPIL_MIN = 15

EYE_AR_THRESH = 0.3  # indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
file = 'Resources/face_landmarks.dat'

left_pupil_data = []
right_pupil_data = []
mean = []

class Eye_Detector:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file)

    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    blob_detector = cv2.SimpleBlobDetector_create(detector_params)

    def __init__(self):
        self.COUNTER = 0
        self.TOTAL = 0 # blink total
        self.le_keypoints = None
        self.re_keypoints = None
        self.left_d = 0.0
        self.right_d = 0.0

    def write_to_csv(self):
        f = open('trial_pupil' + '.csv', 'w')
        with f:
            writer = csv.writer(f)

            writer.writerow(['left_pupil', 'right_pupil', 'mean'])
            for i in range(len(left_pupil_data)):
                writer.writerow([left_pupil_data[i], right_pupil_data[i], mean[i]])
            print('csv done.' + f.name)

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
        keypoints = self.blob_detector.detect(blur)

        return keypoints

    def blob_process_both_eyes(self, left_eye, right_eye, start_thres):
        left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

        if start_thres == 0:
            end_thres = 130
        else:
            end_thres = start_thres+10
        for thres in range(start_thres-10, end_thres):  # start_thres +- 10
            left_keypoints = self.get_keypoints(left_gray, thres)
            right_keypoints = self.get_keypoints(right_gray, thres)
            if len(left_keypoints) > 0 and len(right_keypoints) > 0:
                return thres, left_keypoints, right_keypoints
        return None, None, None

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # vertical
        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])  # horizontal
        ear = (A + B) / (2.0 * C)
        return ear

    def show_text(self, frame, ear, start_t):
        cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Left Pupil: {:.3f}'.format(self.left_d), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Right Pupil: {:.3f}'.format(self.right_d), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Time {:2f}'.format(time.time() - start_t), (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    def run(self):
        current = -1
        # fig, ax = plt.subplots()
        # legend_element = [mpatches.Patch(color='blue', label='Left pupil', linestyle='-', linewidth=0.5),
        #                   mpatches.Patch(color='green', label='Right pupil', linestyle='-', linewidth=0.5),
        #                   mpatches.Patch(color='red', label='Mean', linestyle='--', linewidth=0.5)]
        # ax.legend(handles=legend_element, loc='upper_left')
        # ax.set_ylim(bottom=0, top=15)
        # ax.set_title('Pupil Dilation')

        cap = cv2.VideoCapture(0)
        start_t = time.time()
        threshold = 0
        while True:
            _, frame = cap.read()
            frame = imutils.resize(frame, width=500, height=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract eyes
                left = self.extract_eye(frame, shape, lStart, lEnd)
                right = self.extract_eye(frame, shape, rStart, rEnd)
                # detect pupil
                trial_thres, le_keypoints, re_keypoints = self.blob_process_both_eyes(left, right, threshold)

                if trial_thres != None:
                    threshold = trial_thres
                # print('thres', threshold)

                if le_keypoints and re_keypoints:
                    ld = le_keypoints[0].size
                    rd = re_keypoints[0].size
                    if ld < PUPIL_MIN and rd < PUPIL_MIN:
                        self.left_d = ld
                        self.right_d = rd
                        m = (ld + rd)/2

                        left_pupil_data.append(ld)
                        right_pupil_data.append(rd)
                        mean.append(m)
                        current += 1

                # detecting blink
                leftEye = shape[lStart: lEnd]
                rightEye = shape[rStart: rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                if ear < EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                    self.COUNTER = 0

                self.show_text(frame, ear, start_t)
                # draw graph

            # ax.plot([i for i in range(len(left_pupil_data))], left_pupil_data, '-g')
            # ax.plot([i for i in range(len(right_pupil_data))], right_pupil_data, '-b')
            # ax.plot([i for i in range(len(mean))], mean, '--r', )

            # fig.show()
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.write_to_csv()
                break

        cap.release()
        cv2.destroyAllWindows()
