from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import time


file = 'Resources/face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file)

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.minArea = 12
detector_params.maxArea = 50
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def extract_eye(shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:y + h, x:x + w]
    return roi

def get_keypoints(image, detector, threshold):
        image = imutils.resize(image, width=50, inter=cv2.INTER_CUBIC)
        image = cv2.GaussianBlur(image, (3,3), 1)
        _, thres = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        # dilate = cv2.dilate(thres, (2, 2), iterations=1)
        # erode = cv2.erode(dilate, (1, 1), iterations=1)

        # _, contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        #
        # for cnt in contours:
        #     cv2.drawContours(image, [cnt], -1, (0,0,255), 1)
        #     print(cv2.contourArea(cnt))

        blur = cv2.medianBlur(thres, 1)
        keypoints = detector.detect(blur)

        if keypoints:
            print('FOUND', threshold, keypoints[0].size)
            image_k = cv2.drawKeypoints(image, keypoints, image, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('image', image_k)
            return keypoints[0].size

        stack = np.hstack((image, thres, blur))
        cv2.imshow('stack', stack)
        return None

def blob_process(eye, detector):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    optimal = []

    for thres in range(0, 130):  # 0 - 255
        area = get_keypoints(gray, detector, thres)
        if area:
            optimal.append((thres, area))
            pass
        if area is None and len(optimal) > 0:
            break
    if len(optimal) > 0:
        optimal = sorted(optimal, key=lambda x: x[1], reverse=True)
        print('optimal', optimal)
        return optimal[0]
    return None

def blob_process_both_eyes(left_eye, right_eye, detector):
    left_optimal = blob_process(left_eye, detector)
    right_optimal = blob_process(right_eye, detector)

    if left_optimal and right_optimal:
        print('left', left_optimal, 'right', right_optimal)
        return left_optimal, right_optimal

    return None, None

frame = cv2.imread('test.jpg')
frame = imutils.resize(frame, width=500, height=500, inter=cv2.INTER_CUBIC)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left = extract_eye(shape, lStart, lEnd)
        right = extract_eye(shape, rStart, rEnd)

        result_left, result_right = blob_process_both_eyes(left, right, blob_detector)

    # cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
