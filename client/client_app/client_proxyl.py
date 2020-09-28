import socket
from imutils import face_utils
import imutils
import cv2
import time
import dlib
import argparse
import os.path as path
import Tkinter
import tkMessageBox
from scipy.spatial import distance as dist
import numpy as np
import pickle
from Task import Task

face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_file)

global pre_pupil_data
pre_pupil_data_left = []
pre_pupil_data_right = []

EYE_AR_THRESH = 0.21  # default threshold for indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.minArea = 45
detector_params.maxArea = 200
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

global run
run = True

def extract_eye(frame, shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:(y + h), x:(x + w)]
    roi = roi[int(h * 0.05): (h), int(w * 0.2): int(w * 0.9)]
    return roi


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # vertical
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear


def show_text(client, frame, ear, time):
    cv2.putText(frame, "Blinks: {}".format(client.blink_count), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    cv2.putText(frame, 'Left Pupil: {:.5f}'.format(client.left_pupil / 3), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(frame, 'Right Pupil: {:.5f}'.format(client.right_pupil / 3), (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    cv2.putText(frame, 'Time {:2f}'.format(time - client.start_time), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


def terminate_detect():
    global run
    run = False


###################### main detection ######################
def image_preprocess(image, threshold):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = cv2.boundingRect(image)
    image = imutils.resize(image, width=int(w * 3), height=int(h * 3), inter=cv2.INTER_CUBIC)
    _, thres = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(thres, 1)
    return blur, image, w, h


def get_keypoints(client, image, threshold, side):
    if side == 0:
        min_x = client.left_x[0] if client.left_x else 0
        max_x = client.left_x[1] if client.left_x else 20
        min_y = client.left_y[0] if client.left_y else 0
        max_y = client.left_y[1] if client.left_y else 20

    if side == 1:
        min_x = client.right_x[0] if client.right_x else 0
        max_x = client.right_x[1] if client.right_x else 20
        min_y = client.right_y[0] if client.right_y else 0
        max_y = client.right_y[1] if client.right_y else 20

    img, _, _, _ = image_preprocess(image, threshold)
    keypoints = blob_detector.detect(img)

    if keypoints:
        if keypoints[0].pt[0] >= min_x and keypoints[0].pt[0] <= max_x:
            if keypoints[0].pt[1] >= min_y and keypoints[0].pt[1] <= max_y:
                # image_k = cv2.drawKeypoints(image, keypoints, image, (0, 0, 255),
                #                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow('image', image_k)
                # print(side, (x, y, w, h), keypoints[0].pt, keypoints[0].size)
                return keypoints[0].size  # diameter

    # stack = np.hstack((image, thres, blur))
    # cv2.imshow('stack', stack)
    return None


def blob_process(client, eye, side):
    optimal = []

    if side == 0:  # 0 - left side
        start_threshold = client.left_threshold[0] #if client.left_threshold is not None else 10
        end_threshold = client.left_threshold[1] #if client.left_threshold is not None else 130
    if side == 1:  # 1 - right side
        start_threshold = client.right_threshold[0] #if client.right_threshold is not None else 10
        end_threshold = client.right_threshold[1] #if client.right_threshold is not None else 130

    for thres in range(start_threshold, end_threshold):
        area = get_keypoints(client=client, image=eye, threshold=thres, side=side)
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


def blob_process_both_eyes(client, left_eye, right_eye):
    left_optimal = blob_process(client, left_eye, 0)
    right_optimal = blob_process(client, right_eye, 1)

    # # print('left', left_optimal, 'right', right_optimal)
    # if left_optimal is not None or right_optimal is not None:
    #     return left_optimal, right_optimal

    return left_optimal, right_optimal


################### calibration ############################
def handle_blink(shape, frame, show):
    # detecting blink
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    if show:
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    ear = (leftEAR + rightEAR) / 2.0
    return ear


def calib_eye(left, right, eye, calib_count):
    images = []
    max_thres = 0

    for threshold in range(0, 130):
        result = get_keypoints_calib(eye, threshold)
        if result:
            max_thres = max(max_thres, threshold)
            images.append(result)
        # if result is None and threshold > max_thres + 10 and len(images) > 0:
        #     break

    for i in images:
        cv2.imshow('image', i[0])

        if cv2.waitKey(20000) & 0xFF == ord('y'):
            cv2.destroyWindow('image')
            calib_count -= 1
            tkMessageBox.showinfo('Sample count Left', '{} left'.format(calib_count))

            if left:
                pre_pupil_data_left.append(i[1:])
                if len(pre_pupil_data_left) >= 5:
                    left = False
                    right = True
                    return left, right, 5
            if right:
                pre_pupil_data_right.append(i[1:])
                if len(pre_pupil_data_right) >= 5:
                    left = False
                    right = False
                    return left, right, 5

        cv2.destroyWindow('image')
    return None, None, calib_count


def get_keypoints_calib(image, threshold):
    img, gray, w, h = image_preprocess(image, threshold)
    keypoints = blob_detector.detect(img)
    if keypoints:
        if keypoints[0].pt[0] >= w * 0.7 and keypoints[0].pt[0] >= h * 0.33 and keypoints[0].size < 10:
            image_k = cv2.drawKeypoints(gray, keypoints, gray, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (image_k, threshold, keypoints[0].pt, keypoints[0].size)

    return None


def read_control():
    control_file = path.join(path.dirname(__file__), 'control.txt')
    with open(control_file, 'r') as f:
        lines = f.readline(1)
        f.close()
        return lines[0]


def read_error_list():
    file = path.join(path.dirname(__file__), 'error_list.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
        f.close()
    return lines


def send_data(client, socket, blink=False):
    if blink:
        data = Task(client.time, client.left_pupil / 3 if client.left_pupil != 0 else 0,
                    client.right_pupil / 3 if client.right_pupil != 0 else 0, blink=client.blink_count, error=0)
        obj = pickle.dumps(data)
        socket.sendall(b'[D]' + obj)
    else:
        if client.left_pupil == 0 and client.right_pupil == 0:
            return
        else:
            data = Task(client.time, client.left_pupil / 3 if client.left_pupil != 0 else 0,
                        client.right_pupil / 3 if client.right_pupil != 0 else 0, blink=np.nan, error=0)
            obj = pickle.dumps(data)
            socket.sendall(b'[D]' + obj)
    time.sleep(0.5)


##################### main program #########################
def run_with_server(HOST='localhost', PORT=8080, client=None):
    try:
        print(client)
        ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = ServerSocket.connect((HOST, PORT))
        print('Connection Established')
        run = '1'

        while True:

            if run == '0':
                break

            cap = cv2.VideoCapture(0)

            client.socket = ServerSocket
            blink_start_t = time.time()
            blink_counter = 0

            while (run == '1'):

                run = read_control()

                _, frame = cap.read()
                frame = imutils.resize(frame, width=500, height=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect face
                rects = face_detector(gray, 0)

                for rect in rects:
                    # detect sense organs
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # get eyes
                    left_eye = extract_eye(frame, shape, lStart, lEnd)
                    right_eye = extract_eye(frame, shape, rStart, rEnd)
                    # detect pupil size
                    left_size, right_size = blob_process_both_eyes(client, left_eye, right_eye)

                    client.time = time.time()
                    client.left_pupil = left_size if left_size is not None else 0
                    client.right_pupil = right_size if right_size is not None else 0

                    # print(client.left_pupil, client.right_pupil, 'data updated!!')

                    # detecting blink
                    leftEye = shape[lStart: lEnd]
                    rightEye = shape[rStart: rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    ear = (leftEAR + rightEAR) / 2.0
                    # print(ear)

                    if ear < EYE_AR_THRESH:
                        blink_counter += 1
                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if blink_counter >= EYE_AR_CONSEC_FRAMES:
                            client.blink_count += 1
                            # print('[*BLINK*] blink_count=', client.blink_count)
                        # reset the eye frame counter
                        blink_counter = 0

                    if time.time() - blink_start_t >= 10:
                        send_data(client=client, socket=ServerSocket, blink=True)
                        client.blink_count = 0
                        blink_start_t = time.time()
                    elif not (client.left_pupil == 0 and client.right_pupil == 0):
                        send_data(client=client, socket=ServerSocket)

            errors = read_error_list()
            obj = pickle.dumps(errors)
            ServerSocket.sendall(b'[E]' + obj)


    except socket.error as err:
        print(err)

    finally:
        ServerSocket.close()


def run_standalone(client):
    print(client)

    root = Tkinter.Tk()
    root.withdraw()

    tkMessageBox.showinfo('Calibration', 'Please open your eyes and stay for 1 second.\nWhen you ready, press Any Key')

    cap = cv2.VideoCapture(0)
    client.start_time = time.time()
    blink_counter = 0

    MAX_EAR = 0
    EAR_UNCHANGED = 0

    calib = 0
    left = True
    right = False
    show_hull = True
    calib_count = 5

    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=500, height=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face
        rects = face_detector(gray, 0)
        for rect in rects:
            # detect sense organs
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # detect blink
            ear = handle_blink(shape, frame, show_hull)

            if EAR_UNCHANGED < 3:
                MAX_EAR = max(MAX_EAR, ear)
                if (ear // 0.001) == (MAX_EAR // 0.001):
                    EAR_UNCHANGED += 1
                else:
                    EAR_UNCHANGED = 0

            elif EAR_UNCHANGED == 3:
                global EYE_AR_THRESH
                EYE_AR_THRESH = MAX_EAR * 0.7
                EAR_UNCHANGED += 1  # make it to 5 so it won't run this line again
                client.calib_blink = False  # calibration for blink end
                tkMessageBox.showinfo('Calibration', 'Blink SET! {}'.format(EYE_AR_THRESH))
                show_hull = False
                calib = 1
                # print(EAR_UNCHANGED, EYE_AR_THRESH)
                tkMessageBox.showinfo('Calibration',
                                      'Please select the image that circled your pupil correctly.\nWhen you ready, press Any Key')

            if ear < EYE_AR_THRESH:
                blink_counter += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    client.blink_count += 1
                # reset the eye frame counter
                blink_counter = 0

            # get eyes
            if calib == 1 and left:
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                r1, r2, calib_count = calib_eye(left, right, left_eye, calib_count)
                if r1 is not None and r2 is not None:
                    left = r1
                    right = r2  # finish left eye
                    tkMessageBox.showinfo('Calibration', 'Left eye done!')

            elif calib == 1 and right:
                right_eye = extract_eye(frame, shape, rStart, rEnd)
                r1, r2, calib_count = calib_eye(left, right, right_eye, calib_count)
                if r1 is not None and r2 is not None:
                    left = r1
                    right = r2  # finish right eye
                    calib = 2  # finish eye calibration
                    client.set_filtering(pre_left=pre_pupil_data_left, pre_right=pre_pupil_data_right)
                    tkMessageBox.showinfo('Calibration', 'Right eye done!\nCalibration All Done.')

            if calib == 2:
                # detect pupil size
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                right_eye = extract_eye(frame, shape, rStart, rEnd)
                left_size, right_size = blob_process_both_eyes(client, left_eye, right_eye)

                client.time = time.time()
                if left_size is not None:
                    client.left_pupil = left_size
                if right_size is not None:
                    client.right_pupil = right_size

            show_text(client, frame, ear, time.time())

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.reset()
    return client

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='run method')
    my_parser.add_argument('--method',
                           help='server/standalone')
    my_parser.add_argument('--host',
                           help='host address to connect')

    args = my_parser.parse_args()
    if args.method == 'server':
        if args.host:
            run_with_server(HOST=args.host)
        else:
            run_with_server()
    elif args.method == 'standalone':
        run_standalone()
