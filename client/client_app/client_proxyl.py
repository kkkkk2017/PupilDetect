from imutils import face_utils
import imutils
import cv2
import time
import dlib
import argparse
import os.path as path
from tkinter import *
from tkinter import messagebox
from scipy.spatial import distance as dist
import numpy as np
import pickle
import csv
from Task import Task
from client import Client

face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
predictor = dlib.shape_predictor(face_file)
face_detector = dlib.get_frontal_face_detector()

global pre_pupil_data
pre_pupil_data_left = []
pre_pupil_data_right = []
pre_left_threshold = []
pre_right_threshold = []

EYE_AR_THRESH = 0.21  # default threshold for indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# detector_params = cv2.SimpleBlobDetector_Params()
# detector_params.filterByArea = True
# detector_params.minArea = 45
# detector_params.maxArea = 200
# blob_detector = cv2.SimpleBlobDetector_create(detector_params)

data_file = path.join(path.dirname(__file__), 'data.txt')
control_file = path.join(path.dirname(__file__), 'control.txt')

def extract_eye(frame, shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:(y + h), x:(x + w)]
    roi = imutils.resize(roi, width=100, height=75, inter=cv2.INTER_CUBIC)
    return roi


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # vertical
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear


def show_text(client, frame, ear, time):
    cv2.putText(frame, "Blinks: {}".format(client.blink_count), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.putText(frame, 'Left Pupil: {:.5f}'.format(client.current_left_pupil), (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, 'Right Pupil: {:.5f}'.format(client.current_right_pupil), (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.putText(frame, 'Time {:2f}'.format(time - client.start_time), (10, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


###################### main detection ######################
def pupil_preprocess(image, threshold, iris, show=False, calib=False):
    hist = cv2.equalizeHist(image)
    open = cv2.morphologyEx(hist, cv2.MORPH_OPEN, (5, 5))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, (5, 5))
    gaublur = cv2.GaussianBlur(close, (5, 5), 1)
    _, thres = cv2.threshold(gaublur, threshold, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(thres, 5)
    #use findcontours
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if show:
        hstack = np.hstack((hist, open, close))
        cv2.imshow('pupil', hstack)

    if contours is not None:
        # cv2.drawContours(image, contours, 0, (0, 255, 0), 1)
        closest = None
        closet_d = None

        for i in contours:
            (x, y), radius = cv2.minEnclosingCircle(i)

            # calibration uses integer for drawing circle
            if calib:
                x = int(x)
                y = int(y)
                radius = int(radius)

            # if raduis > iris radius, then it is not
            if radius >= iris[2]: continue

            ratio = radius/iris[2]
            # check pupil/iris ratio
            if ratio <= 0.2 or ratio >= 0.60: continue

            dist_x = np.abs(iris[0]-x)
            dist_y = np.abs(iris[1]-y)

            if (np.abs(dist_x) > 2) or np.abs(dist_y) > 2: continue

            d = np.sqrt( np.square(dist_x) + np.square(dist_y) )
            # print('distance: ', d)

            if d > 10: continue
            # print(radius, iris[2], ratio, d)

            if closet_d is None:
                closet_d = d
                closest = (x, y, radius)
            else:
                print(closet_d, closest, 'compare to', d, (x, y, radius))
                if d < closet_d:
                    closest = (x, y, radius)

        return closest

    # using hough circles
    # pupil = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=5, minRadius=0, maxRadius=10)
    #
    # if pupil is not None:
    #     if show:
    #         hstack = np.hstack((hist, open, close))
    #         cv2.imshow('pupil', hstack)
    #
    #     # pupil = np.uint16(np.around(pupil, decimals=3))
    #     pupil = sort_results(pupil[0])
    #     #  size, y, x
    #     x = pupil[0][0]
    #     y = pupil[0][1]
    #     radius = pupil[0][2]
    #
    #     dist_x = np.abs(iris[0]-x)
    #     dist_y = np.abs(iris[1]-y)
    #
    #     if (np.abs(dist_x) <= 2) and (np.abs(dist_y) <= 2):
    #         ratio = radius/iris[2]
    #         if ratio > 0.2 and ratio < 0.72:
    #             return (x, y, radius)

    return None


def iris_preprocess(image):
    blur_iris = cv2.GaussianBlur(image, (5, 5), 1)
    edge = cv2.Canny(blur_iris, 60, 60)
    # cv2.imshow('iris', edge)
    # open = cv2.morphologyEx(edge, cv2.MORPH_OPEN, (3, 3))
    iris = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 5, param1=10, param2=10, minRadius=18, maxRadius=25)
    if iris is not None:
        # iris = np.uint16(np.around(iris, decimals=3))
        iris = sort_results(iris[0])
        # print('iris', iris)
        return iris[0]
    return None


def sort_results(result):
    result = sorted(result, key=lambda x: x[2], reverse=True)
    return result

#detecting both iris and pupil
def iris_pupil(eye, threshold, calib=False):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    iris = iris_preprocess(gray) #(x, y)

    if iris is not None:
        if calib:
            #if calib, reutrn int for circle()
            pupil = pupil_preprocess(gray, threshold, iris, show=True, calib=True)
        else:
            #else, return float
            pupil = pupil_preprocess(gray, threshold, iris)
        #when use findcontours
        if pupil is not None:
            if calib:
                image = np.copy(eye)
                image = cv2.circle(image, (int(iris[0]), int(iris[1])), int(iris[2]), (0, 0, 255), 1)
                # print('pupil', pupil)
                image = cv2.circle(image, (pupil[0], pupil[1]), pupil[2], (0, 0, 255), 1)
                return (image, iris, pupil)
            else:
                return (iris, pupil)

    #when use houghcircle
    # if pupil is not None:
    #     dist_x = np.abs(pupil[0]-iris[0])
    #     dist_y = np.abs(pupil[1]-iris[1])
    #     if (dist_x >= 0 and dist_x <= 2) and (dist_y >= 0 and dist_y <= 2):
    #         if calib:
    #             image = np.copy(eye)
    #             image = cv2.circle(image, (iris[0], iris[1]), iris[2], (0, 0, 255), 1)
    #             image = cv2.circle(image, (pupil[0], pupil[1]), pupil[2], (0, 0, 255), 1)
    #             return (image, iris, pupil)
    #         else:
    #             return (iris, pupil)

    return None

#validate the keypint from iris_pupil()
def get_keypoints(client=None, image=None, threshold=0, side=None):
    if client is None:
        print('get keypoint: client is None')
        return

    if side == 0:
        min_x = client.left_x[0] if client.left_x else 45
        max_x = client.left_x[1] if client.left_x else 55
        min_y = client.left_y[0] if client.left_y else 13
        max_y = client.left_y[1] if client.left_y else 20
        iris_limit = client.left_iris

    if side == 1:
        min_x = client.right_x[0] if client.right_x else 45
        max_x = client.right_x[1] if client.right_x else 55
        min_y = client.right_y[0] if client.right_y else 13
        max_y = client.right_y[1] if client.right_y else 20
        iris_limit = client.right_iris #[(min_x, max_x), (min_y, max_y), (min_size, max_size)]

    result = iris_pupil(image, threshold)
    if result is not None:
        iris, pupil = result
        # print('getting result and check iris', iris, iris_limit)
        # if iris[0] >= iris_limit[0][0] and iris[0] <= iris_limit[0][1]: #check x
        #     if iris[1] >= iris_limit[1][0] and iris[1] <= iris_limit[1][1]: #check y
        if iris[2] >= iris_limit[2][0] and iris[2] <= iris_limit[2][1]: #check size
            #check pupil
            # print('check pupil', side, pupil, min_x, max_x, min_y, max_y)

            if pupil[0] >= min_x and pupil[0] <= max_x: #check x
                if pupil[1] >= min_y and pupil[1] <= max_y: #check y
                    return (iris[2], pupil[2]) #only need the radius
    return None


def eye_process(client, eye, side):
    if client is None:
        print('get keypoint: client is None')
        return

    if side == 0:
        min_pupil_x = client.left_x[0] if client.left_x else 45
        max_pupil_x = client.left_x[1] if client.left_x else 55
        min_pupil_y = client.left_y[0] if client.left_y else 13
        max_pupil_y = client.left_y[1] if client.left_y else 20
        iris_limit = client.left_iris
        start_thres = client.left_threshold[0] if client.left_threshold else 0
        end_thres = client.left_threshold[1] if client.left_threshold else 130

    if side == 1:
        min_pupil_x = client.right_x[0] if client.right_x else 45
        max_pupil_x = client.right_x[1] if client.right_x else 55
        min_pupil_y = client.right_y[0] if client.right_y else 13
        max_pupil_y = client.right_y[1] if client.right_y else 20
        iris_limit = client.right_iris # [(min_x, max_x), (min_y, max_y), (min_size, max_size)]
        start_thres = client.right_threshold[0] if client.right_threshold else 0
        end_thres = client.right_threshold[1] if client.right_threshold else 130

    iris_mean_x = np.mean(iris_limit[0])
    iris_mean_y = np.mean(iris_limit[1])
    iris_mean_size = np.mean(iris_limit[2])

    pupil_mean_x = np.mean((min_pupil_x, max_pupil_x))
    pupil_mean_y = np.mean((min_pupil_y, max_pupil_y))

    result_set = {}
    dist_set = []

    for thres in range(start_thres, end_thres):
        result = iris_pupil(eye, thres)
        if result is not None:
            # print('result', result)
            iris, pupil = result
            result_set.update({thres: result})
            value_i = (distance(iris[0], iris_mean_x, iris[1], iris_mean_y), np.abs(iris[2]-iris_mean_size) )
            value_p = distance(pupil[0], pupil_mean_x, pupil[1], pupil_mean_y)
            dist_set.append((thres, value_i, value_p))

    if len(dist_set) != 0:
        # print(dist_set)
        dist_set = sorted(dist_set, key=lambda x: (x[1][1], x[1][0], x[2])) #sort by iris first
        # print(dist_set)
        result = result_set.get(dist_set[0][0])
        # print('result=', result)
        return result[0], result[1] #only return the coordinate and size

    return None

def distance(x1, x2, y1, y2):
    result = np.square((x2-x1)) + np.square((y2-y1))
    result = np.sqrt(result)
    return result


def process_both_eyes(client, left_eye, right_eye):
    left_optimal = eye_process(client, left_eye, 0)
    right_optimal = eye_process(client, right_eye, 1)
    return left_optimal, right_optimal


################### calibration ############################
def handle_blink(shape, frame, show=False):
    # detecting blink
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # if show:
    # leftEyeHull = cv2.convexHull(leftEye)
    # rightEyeHull = cv2.convexHull(rightEye)
    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    ear = (leftEAR + rightEAR) / 2.0
    return np.around(ear, decimals=2)

def calib_eye(left, right, eye, calib_count):
    result_lst = []

    for threshold in range(0, 140):
        # result = get_keypoints(image=eye, threshold=threshold, calib=True)
        result = iris_pupil(eye, threshold, True)

        if result:
            result_lst.append((threshold, result))
        # if result is None and threshold > max_thres + 10 and len(images) > 0:
        #     break
    # print(len(result_lst))
    for (thres, i) in result_lst:
        cv2.imshow('image', i[0])

        if cv2.waitKey(20000) & 0xFF == ord('y'):
            # print('selected', i[1:])
            cv2.destroyWindow('image')
            calib_count -= 1
            if calib_count % 5 == 0:
                messagebox.showinfo('Sample count Left', '{} left'.format(calib_count))

            if left:
                pre_pupil_data_left.append((i[1], i[2]))
                pre_left_threshold.append(thres)
                if len(pre_pupil_data_left) >= 15:
                    left = False
                    right = True
                    return left, right, 15
            if right:
                pre_pupil_data_right.append((i[1], i[2]))
                pre_right_threshold.append(thres)
                if len(pre_pupil_data_right) >= 15:
                    left = False
                    right = False
                    return left, right, 15

        cv2.destroyWindow('image')
    return None, None, calib_count


def read_control():
    with open(control_file, 'r') as f:
        lines = f.readline(1)
        f.close()
    return lines[0]

def store_data(task):
    with open(data_file, 'a') as f:
        if task:
            for i in task.to_List():
                f.write(str(i)+',')
            f.write('\n')
    f.close()

def update_data(client, left_result, right_result):
    # print(left_result, right_result)

    if left_result is not None:
        left_iris, left_pupil = left_result
        # print('left_iris',left_iris)
        # print('left_pupil', left_pupil)
        # print(left_size, right_size, 'data updated!!')
        client.current_left_iris_x = left_iris[0]
        client.current_left_iris_y = left_iris[1]
        client.current_left_iris = left_iris[2]

        client.current_left_pupil_x = left_pupil[0]
        client.current_left_pupil_y = left_pupil[1]
        client.current_left_pupil = left_pupil[2]

    if right_result is not None:
        right_iris, right_pupil = right_result

        client.current_right_iris_x = right_iris[0]
        client.current_right_iris_y = right_iris[1]
        client.current_right_iris = right_iris[2]

        client.current_right_pupil_x = right_pupil[0]
        client.current_right_pupil_y = right_pupil[1]
        client.current_right_pupil = right_pupil[2]

    return client

##################### main program #########################
def run(client=None):
    if client is None:
        client = Client()

    with open(data_file, 'w') as f:
        pass

    run = read_control()
    last_update = time.time()

    while True:
        if run == '0':
            print('run is 0')
            break

        cap = cv2.VideoCapture(0)

        blink_t = time.time() # the time for send the blink (per 10 sec)
        task_t = time.time() #the time for send the task (per 1 sec)
        blink_counter = 0
        # print('video up')

        while (run != '0'):
            # print(run)
            last_t = time.time()
            print(last_t - client.current_time)
            run = read_control()

            _, frame = cap.read()
            frame = imutils.resize(frame, width=1000, height=1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect face
            rects = face_detector(gray, 0)
            # print('get face')
            for rect in rects:
                # detect sense organs
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # print('blink')
                #detect blink
                ear = handle_blink(shape, frame=None, show=False)
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

                # print('eye')
                # get eyes
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                right_eye = extract_eye(frame, shape, rStart, rEnd)

                # detect pupil size
                left_result, right_result = process_both_eyes(client, left_eye, right_eye)
                # print(left_result, right_result)
                # print(left_size, right_size)
                #update time
                client.current_time = time.time()
                client = update_data(client, left_result, right_result)

                if run == 'n':
                    client.blink_count = 0

                if run == 'a':
                    if client.current_time - blink_t >= 10:
                        t = Task(time=client.current_time,
                                 left_pupil=client.current_left_pupil, left_pupil_x=client.current_left_pupil_x,
                                 left_pupil_y=client.current_left_pupil_y,
                                 right_pupil=client.current_right_pupil, right_pupil_x=client.current_right_pupil_x,
                                 right_pupil_y=client.current_right_pupil_y,
                                 left_iris=client.current_left_iris, left_iris_x=client.current_left_iris_x,
                                 left_iris_y=client.current_left_iris_y,
                                 right_iris=client.current_right_iris, right_iris_x=client.current_right_iris_x,
                                 right_iris_y=client.current_right_iris_y,
                                 blink_count=client.blink_count)
                        store_data(t)
                        client.reset_data()
                        blink_t = time.time()
                        task_t = time.time()

                    # elif client.current_time - task_t >= 0.001:
                    if not (client.current_left_pupil == 0 and client.current_right_pupil == 0):
                        t = Task(time=client.current_time,
                                 left_pupil=client.current_left_pupil, left_pupil_x=client.current_left_pupil_x,
                                 left_pupil_y=client.current_left_pupil_y,
                                 right_pupil=client.current_right_pupil, right_pupil_x=client.current_right_pupil_x,
                                 right_pupil_y=client.current_right_pupil_y,
                                 left_iris=client.current_left_iris, left_iris_x=client.current_left_iris_x,
                                 left_iris_y=client.current_left_iris_y,
                                 right_iris=client.current_right_iris, right_iris_x=client.current_right_iris_x,
                                 right_iris_y=client.current_right_iris_y,
                                 blink_count=np.nan)
                        store_data(t)
                        client.reset_data()
                        task_t = time.time()

    cap.release()
    cv2.destroyAllWindows()



def run_standalone(client):

    root = Tk()
    root.withdraw()

    messagebox.showinfo('Calibration', 'Please open your eyes and stay for 1 second.\nWhen you ready, press Any Key')

    cap = cv2.VideoCapture(0)
    client.start_time = time.time()
    blink_counter = 0

    MAX_EAR = 0
    EAR_UNCHANGED = 0

    calib = 0
    left = True
    right = False
    show_hull = True
    calib_count = 15

    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=1000, height=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face
        rects = face_detector(gray, 0)
        for rect in rects:
            # detect sense organs
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # detect blink
            ear = handle_blink(shape, frame, show_hull)
            if EAR_UNCHANGED < 2:
                MAX_EAR = max(MAX_EAR, ear)
                if (ear // 0.001) == (MAX_EAR // 0.001):
                    EAR_UNCHANGED += 1
                else:
                    EAR_UNCHANGED = 0

            elif EAR_UNCHANGED == 2:
                global EYE_AR_THRESH
                EYE_AR_THRESH = MAX_EAR * 0.75
                EAR_UNCHANGED += 1  # make it to 5 so it won't run this line again
                client.calib_blink = False  # calibration for blink end
                messagebox.showinfo('Calibration', 'Blink SET! {}'.format(EYE_AR_THRESH))
                time.sleep(1)
                show_hull = False
                calib = 1
                # print(EAR_UNCHANGED, EYE_AR_THRESH)
                messagebox.showinfo('Calibration',
                                      'Please select the image that circled your pupil correctly by press \'y\', '
                                      'otherwise, anyother key.\nWhen you ready, press Any Key')

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
                    messagebox.showinfo('Calibration', 'Left eye done!')

            elif calib == 1 and right:
                right_eye = extract_eye(frame, shape, rStart, rEnd)
                r1, r2, calib_count = calib_eye(left, right, right_eye, calib_count)
                if r1 is not None and r2 is not None:
                    left = r1
                    right = r2  # finish right eye
                    calib = 2  # finish eye calibration
                    client.set_filtering(pre_left=pre_pupil_data_left, pre_right=pre_pupil_data_right,
                                         left_threshold=pre_left_threshold, right_threshold=pre_right_threshold)
                    messagebox.showinfo('Calibration', 'Right eye done!\nCalibration All Done.')

            #calibration finish
            if calib == 2:
                # detect pupil size
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                right_eye = extract_eye(frame, shape, rStart, rEnd)

                # detect pupil size
                left_result, right_result = process_both_eyes(client, left_eye, right_eye)
                # print(left_size, right_size)
                #update time
                client.current_time = time.time()
                client = update_data(client, left_result, right_result)

                # client.current_time = time.time()
                # if left_size is not None:
                #     client.current_left_iris = left_size[0]
                #     client.current_left_pupil = left_size[1]
                # if right_size is not None:
                #     client.current_right_iris = right_size[0]
                #     client.current_right_pupil = right_size[1]

            show_text(client, frame, ear, time.time())

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.reset_start_time()
    client.reset_data()
    return client

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='run method')
    my_parser.add_argument('--method',
                           help='run/standalone')
    my_parser.add_argument('--host',
                           help='host address to connect')

    args = my_parser.parse_args()
    if args.method == 'standalone':
        run_standalone()
    elif args.method == 'run':
        run()
