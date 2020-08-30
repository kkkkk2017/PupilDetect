from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np


file = 'Resources/face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file)

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def extract_eye(shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:y + h, x:x + w]
    return roi


def get_keypoints(image, detector, threshold):
    image = imutils.resize(image, width=250, inter=cv2.INTER_CUBIC)
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(image, None, iterations=2)
    dilate = cv2.dilate(erode, None, iterations=4)
    blur = cv2.medianBlur(dilate, 5)
    stack = np.hstack((image, erode, dilate, blur))
    cv2.imshow('stack', stack)
    keypoints = detector.detect(blur)
    return keypoints


def blob_process_both_eyes(left_eye, right_eye, detector):
    left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

    for thres in range(0, 160):  # 0 - 255
        left_keypoints = get_keypoints(left_gray, detector, thres)
        right_keypoints = get_keypoints(right_gray, detector, thres)
        if len(left_keypoints) > 0 and len(right_keypoints) > 0:
            return thres, left_keypoints, right_keypoints
    return thres, None, None


frame = cv2.imread('test.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left = extract_eye(shape, lStart, lEnd)
        right = extract_eye(shape, rStart, rEnd)

        le_keypoints = None
        re_keypoints = None

        thres, le_keypoints, re_keypoints = blob_process_both_eyes(left, right, blob_detector)

        if le_keypoints and re_keypoints:
            left_d = le_keypoints[0].size
            right_d = re_keypoints[0].size
            # cv2.drawKeypoints(left, le_keypoints, left, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.drawKeypoints(right, re_keypoints, right, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        leftEye = shape[lStart: lEnd]
        rightEye = shape[rStart: rEnd]

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
