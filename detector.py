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

PUPIL_MIN = 10

EYE_AR_THRESH = 0.3  # indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
file = 'Resources/face_landmarks.dat'


class Eye_Detector:
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file)

    def __init__(self):
        self._left_d = 0.0
        self._right_d = 0.0
        self._COUNTER = 0
        self._TOTAL = 0 # blink total
        self._left_pupil_data = []
        self._right_pupil_data = []
        self._mean = []
        # self.task = Task()

    # create a new empty task and clear the previous data
    def next_task(self):
        # self.task = Task()
        self._left_d = 0.0
        self._right_d = 0.0
        self._COUNTER = 0
        self._TOTAL = 0 # blink total
        self._left_pupil_data = []
        self._right_pupil_data = []
        self._mean = []

    def get_task(self):
        if self.task is not None:
            self.task.left_pupil = self._left_pupil_data
            self.task.right_pupil = self._right_pupil_data
            self.task.mean = self._mean
            self.task.blink_count = self._TOTAL
            self.task.end_time = time.time()
            return self.task

    def extract_eye(self, frame, shape, start, end):
        (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
        roi = frame[y:(y + h), x-1:(x + w)-1]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        return roi

    # def get_keypoints(self, image, threshold):
    #     # image = cv2.GaussianBlur(image, (7,7), 0)
    #     _, thres = cv2.threshold(image, 89, 255, cv2.THRESH_BINARY)
    #     # _, contours, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # if contours:
    #     #     contours = sorted(thres, key=lambda x: cv2.contourArea(x), reverse=True)
    #
    #     # for cnt in contours:
    #     #     cv2.drawContours(image, [cnt], -1, (0,0,255), 3)
    #
    #     erode = cv2.erode(thres, (3, 3), iterations=1)
    #     dilate = cv2.dilate(erode, (1, 1), iterations=3)
    #     blur = cv2.medianBlur(dilate, 1)
    #
    #     keypoints = self.blob_detector.detect(blur)
    #
    #     stack = np.hstack((image, thres, erode, dilate, blur))
    #     cv2.imshow('stack', stack)
    #     if keypoints:
    #         print(keypoints[0].size)
    #     return keypoints
    #
    # def blob_process_both_eyes(self, left_eye, right_eye, start_thres, fix=False):
    #     result_thres = None
    #     result_le = None
    #     result_re = None
    #
    #     left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    #     right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
    #
    #     if start_thres == 0:
    #         end_thres = 130
    #     elif not fix:
    #         start_thres = start_thres-2
    #         end_thres = start_thres+5
    #     elif fix:
    #         end_thres = start_thres+1
    #     else:
    #         pass
    #
    #     for thres in range(start_thres, end_thres):  # start_thres +- 2
    #         left_keypoints = self.get_keypoints(left_gray, thres)
    #         # right_keypoints = self.get_keypoints(right_gray, thres)
    #         print(thres, left_keypoints)
    #         # if len(left_keypoints) > 0 and len(right_keypoints) > 0:
    #         #     le = left_keypoints[0].size
    #         #     re = right_keypoints[0].size
    #         #     print(left_keypoints[0].size)
    #         #     if le < PUPIL_MIN and re < PUPIL_MIN:
    #         #         result_thres = thres
    #         #         result_le = le
    #         #         result_re = re
    #         #         break
    #
    #     return result_thres, result_le, result_re

    def get_keypoints(self, image, detector, threshold):
        image = imutils.resize(image, width=50, inter=cv2.INTER_CUBIC)
        # image = cv2.GaussianBlur(image, (3, 3), 1)
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

        print('left', left_optimal, 'right', right_optimal)
        if left_optimal is not None and right_optimal is not None:
            return left_optimal, right_optimal

        return None, None

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # vertical
        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])  # horizontal
        ear = (A + B) / (2.0 * C)
        return ear

    def show_text(self, frame, ear, start_t):
        # cv2.putText(frame, "Blinks: {}".format(self._TOTAL), (10, 30),
        # #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        # cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Left Pupil: {:.3f}'.format(self._left_d), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, 'Right Pupil: {:.3f}'.format(self._right_d), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        # cv2.putText(frame, 'Time {:2f}'.format(time.time() - start_t), (10, 350),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    def write_to_csv(self):
        f = open('trial_pupil_bigrange' + '.csv', 'w')
        with f:
            writer = csv.writer(f)

            writer.writerow(['left_pupil', 'right_pupil', 'mean'])
            for i in range(len(self._left_pupil_data)):
                writer.writerow([self._left_pupil_data[i], self._right_pupil_data[i], self._mean[i]])
            print('csv done ' + f.name)

    def nothing(self, x):
        pass

    def run_with_trackbar(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Pupil Detector')
        # cv2.createTrackbar('threshold', 'Pupil Detector', 0, 255, self.nothing)
        threshold_left = 0
        threshold_right = 0
        start_t = time.time()
        while True:
            _, frame = cap.read()
            frame = imutils.resize(frame, width=500, height=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.face_detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract eyes
                left = self.extract_eye(frame, shape, lStart, lEnd)
                right = self.extract_eye(frame, shape, rStart, rEnd)
                # detect pupil
                # threshold = cv2.getTrackbarPos('threshold', 'Pupil Detector')

                le_size, re_size = self.blob_process_both_eyes(left, right, threshold_left, threshold_right)

                if le_size is not None and re_size is not None:
                    print('append')
                    threshold_left = le_size[0]
                    threshold_right = re_size[0]

                    self._left_d = le_size[1]
                    self._right_d = re_size[1]
                    self._left_pupil_data.append(le_size[1])
                    self._right_pupil_data.append(re_size[1])

                    m = np.mean((le_size[1], re_size[1]))
                    self._mean.append(m)

                # self.show_text(frame, 0, start_t)

            cv2.imshow('Pupil Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.write_to_csv()
                break


        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        current = -1
        fig, ax = plt.subplots()
        legend_element = [mpatches.Patch(color='blue', label='Left pupil', linestyle='-', linewidth=0.5),
                          mpatches.Patch(color='green', label='Right pupil', linestyle='-', linewidth=0.5),
                          mpatches.Patch(color='red', label='Mean', linestyle='--', linewidth=0.5)]
        ax.legend(handles=legend_element, loc='upper_left')
        ax.set_ylim(bottom=0, top=15)
        ax.set_title('Pupil Dilation')

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
                        self._left_d = ld
                        self._right_d = rd
                        m = (ld + rd)/2

                        self._left_pupil_data.append(ld)
                        self._right_pupil_data.append(rd)
                        self._mean.append(m)
                        current += 1

                        # # draw graph
                        # ax.scatter([current], [ld], color='green', linewidths=0.2)
                        # ax.scatter([current], [rd], color='blue', linewidths=0.2)
                        # ax.scatter([current], [m], color='red', linewidths=0.2)
                        # fig.canvas.draw()

                # detecting blink
                leftEye = shape[lStart: lEnd]
                rightEye = shape[rStart: rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                if ear < EYE_AR_THRESH:
                    self._COUNTER += 1
                else:
                    if self._COUNTER >= EYE_AR_CONSEC_FRAMES:
                        self._TOTAL += 1
                    self._COUNTER = 0

                self.show_text(frame, ear, start_t)
                # draw graph

            # ax.plot([i for i in range(len(self._left_pupil_data))], self._left_pupil_data, '-g')
            # ax.plot([i for i in range(len(self._right_pupil_data))], self._right_pupil_data, '-b')
            # ax.plot([i for i in range(len(self._mean))], self._mean, '--r', )

            # fig.show()
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


dect =Eye_Detector()
dect.run_with_trackbar()