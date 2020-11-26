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
from Task import Task
from client import Client

face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
predictor = dlib.shape_predictor(face_file)
face_detector = dlib.get_frontal_face_detector()

global pre_pupil_data
pre_pupil_data_left = []
pre_pupil_data_right = []
pre_iris_data_left = []
pre_iris_data_right = []
pre_left_pupil_threshold = []
pre_right_pupil_threshold = []
pre_left_iris_threshold = []
pre_right_iris_threshold = []

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
    # use findcontours
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            ratio = radius / iris[2]
            # check pupil/iris ratio
            if ratio <= 0.2 or ratio >= 0.60: continue

            dist_x = np.abs(iris[0] - x)
            dist_y = np.abs(iris[1] - y)

            if (np.abs(dist_x) > 2) or np.abs(dist_y) > 2: continue

            d = np.sqrt(np.square(dist_x) + np.square(dist_y))
            # print('distance: ', d)

            if d > 10: continue

            if closet_d is None:
                closet_d = d
                closest = (x, y, radius)
            else:
                if d < closet_d:
                    closest = (x, y, radius)

        return closest

    return None

def iris_preprocess(eye, threshold, calib=False, iris_size=None):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    blur_iris = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thres = cv2.threshold(blur_iris, threshold, 255, cv2.THRESH_BINARY_INV)
    open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, (7, 7), 2)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, (7, 7), 2)
    blur = cv2.medianBlur(close, 5)
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        result = []
        for index, i in enumerate(contours):
            (x, y), radius = cv2.minEnclosingCircle(i)
            if calib:
                if radius < 20 or radius > 30:
                    continue
            else:
                if radius < (iris_size - 0.5) or radius > (iris_size + 0.5):
                    continue

            result.append((x, y, radius))

        if len(result) > 0:
            result = sorted(result, key=lambda x: x[2])

            if calib:
                iris = result[0]
                image = np.copy(eye)
                image = cv2.circle(image, (int(iris[0]), int(iris[1])), int(iris[2]), (0, 255, 0), 1)
                return (image, iris)

            return result[0]

    return None


def sort_results(result):
    result = sorted(result, key=lambda x: x[2], reverse=True)
    return result


def eye_process(client, eye, side):
    if client is None:
        print('get keypoint: client is None')
        return

    if side == 0: #left
        iris_size = client.left_iris_size
        start_thres = client.left_pupil_threshold[0] if client.left_pupil_threshold else 0
        end_thres = client.left_pupil_threshold[1] if client.left_pupil_threshold else 130

    if side == 1:
        iris_size = client.right_iris_size
        start_thres = client.right_pupil_threshold[0] if client.right_pupil_threshold else 0
        end_thres = client.right_pupil_threshold[1] if client.right_pupil_threshold else 130

    result_set = {}
    dist_set = []

    iris = find_iris(side, client, eye)

    if iris is None: return None

    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    for thres in range(start_thres, end_thres):
        pupil = pupil_preprocess(image=gray, threshold=thres, iris=iris)
        if pupil is not None:
            result_set.update({thres: (iris, pupil)})
            value_i = np.abs(iris[2] - iris_size)
            value_p = distance(pupil[0], iris[0], pupil[1], iris[1]) #the distance between iris center and pupil center
            dist_set.append((thres, value_i, value_p))

    if len(dist_set) != 0:
        # print(dist_set)
        dist_set = sorted(dist_set, key=lambda x: (x[1], x[2]))  # sort by iris first
        # print(dist_set)
        result = result_set.get(dist_set[0][0])
        # print('result=', result)
        return result[0], result[1]  # only return the coordinate and size

    return None


def distance(x1, x2, y1, y2):
    result = np.square((x2 - x1)) + np.square((y2 - y1))
    result = np.sqrt(result)
    return result


def process_both_eyes(client, left_eye, right_eye):
    left_optimal = eye_process(client, left_eye, 0)
    right_optimal = eye_process(client, right_eye, 1)
    return left_optimal, right_optimal


################### calibration ############################
def handle_blink(shape):
    # detecting blink
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return np.around(ear, decimals=2)

def calib_iris(side, eye, calib_count):
    result_lst = []

    for threshold in range(0, 140):
        result = iris_preprocess(eye, threshold, True)

        if result:
            result_lst.append((threshold, result))

    for (thres, i) in result_lst:
        cv2.imshow('image', i[0])

        if cv2.waitKey(20000) & 0xFF == ord('y'):
            cv2.destroyWindow('image')
            calib_count -= 1
            if calib_count % 5 == 0:
                messagebox.showinfo('Sample count Left', '{} left'.format(calib_count))

            if side==0:
                pre_iris_data_left.append(i[1])
                # print(pre_iris_data_left)
                pre_left_iris_threshold.append(thres)
                if len(pre_iris_data_left) >= 10:
                    return True, 10

            elif side==1:
                pre_iris_data_right.append(i[1])
                pre_right_iris_threshold.append(thres)
                if len(pre_iris_data_right) >= 10:
                    return True, 10

        else:
            cv2.destroyWindow('image')
    return False, calib_count

def find_iris(side, client, eye):
    current_iris = None
    # find iris first
    if side == 0:
        start_threshold = client.left_iris_threshold[0]
        end_threshold = client.left_iris_threshold[1]
        iris_size = client.left_iris_size
    elif side == 1:
        start_threshold = client.right_iris_threshold[0]
        end_threshold = client.right_iris_threshold[1]
        iris_size = client.right_iris_size

    result_lst = []
    for threshold in range(start_threshold, end_threshold):
        result_iris = iris_preprocess(eye, threshold, iris_size=iris_size)
        if result_iris is not None:
            # if result_iris[2]//1 == iris_size//1:
            result_lst.append(result_iris)

    if len(result_lst) > 0:
        result_lst = sorted(result_lst, key=lambda x: abs(x[2] - iris_size))
        # print('iris result list', result_lst)
        current_iris = result_lst[0]
    # print('current iris', current_iris)
    return current_iris

def calib_eye(client, side, eye, calib_count):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    result_lst = []
    current_iris = find_iris(side, client, eye)

    if current_iris is not None:
        for threshold in range(0, 130):
            pupil = pupil_preprocess(image=gray, iris=current_iris, threshold=threshold, show=True, calib=True)
            if pupil is not None:
                image = np.copy(eye)
                image = cv2.circle(image, (int(current_iris[0]), int(current_iris[1])), int(current_iris[2]), (0, 0, 255), 1)
                # print('pupil', pupil)
                image = cv2.circle(image, (pupil[0], pupil[1]), pupil[2], (0, 0, 255), 1)
                result_lst.append((threshold, (image, pupil)))

    for (thres, i) in result_lst:
        cv2.imshow('image', i[0])

        if cv2.waitKey(20000) & 0xFF == ord('y'):
            cv2.destroyWindow('image')
            calib_count -= 1
            if calib_count % 5 == 0:
                messagebox.showinfo('Sample count Left', '{} left'.format(calib_count))

            if side==0:
                pre_pupil_data_left.append(i[1])
                pre_left_pupil_threshold.append(thres)
                if len(pre_pupil_data_left) >= 10:
                    return True, 10
            elif side==1:
                pre_pupil_data_right.append(i[1])
                pre_right_pupil_threshold.append(thres)
                if len(pre_pupil_data_right) >= 10:
                    return True, 10
        else:
            cv2.destroyWindow('image')

    return False, calib_count


def read_control():
    with open(control_file, 'r') as f:
        lines = f.readline(1)
        f.close()
    return lines[0]


def store_data(task):
    with open(data_file, 'a') as f:
        if task:
            for i in task.to_List():
                f.write(str(i) + ',')
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

        blink_t = time.time()  # the time for send the blink (per 10 sec)
        task_t = time.time()  # the time for send the task (per 1 sec)
        blink_counter = 0
        # print('video up')

        while (run != '0'):
            # print(run)
            last_t = time.time()
            # print(last_t - client.current_time)
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
                # detect blink
                ear = handle_blink(shape)
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
                # update time
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
                    else:
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
    calib_count = 10
    current_calib_iris = True
    current_calib_pupil = False

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
            ear = handle_blink(shape)
            if calib == 0:
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
                    messagebox.showinfo('Calibration', 'Blink SET! {} Please then blink 3 times'.format(EYE_AR_THRESH))
                    time.sleep(1)

            if client.blink_count == 3:
                if calib == 0:
                    calib = 1
                    messagebox.showinfo('Calibration',
                                    'Please select the image that circled your iris correctly by press \'y\', '
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

            if calib == 1: #calib for iris first
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                if current_calib_iris:
                    done, calib_count = calib_iris(side=0, eye=left_eye, calib_count=calib_count)
                    if done:
                        client.set_iris_fitering(pre_iris_data_left, pre_left_iris_threshold, if_left=True)
                        messagebox.showinfo('Calibration', 'Next, please select the image that circled your pupil correctly')
                        current_calib_iris = False
                        current_calib_pupil = True

                elif current_calib_pupil:
                    done, calib_count = calib_eye(client, side=0, eye=left_eye, calib_count=calib_count)
                    if done:
                        messagebox.showinfo('Calibration', 'Left eye done!')
                        calib = 2
                        current_calib_iris = True
                        current_calib_pupil = False

            elif calib == 2:
                right_eye = extract_eye(frame, shape, rStart, rEnd)
                if current_calib_iris:
                    done, calib_count = calib_iris(side=1, eye=right_eye, calib_count=calib_count)
                    if done:
                        client.set_iris_fitering(pre_iris_data_right, pre_right_iris_threshold, if_left=False)
                        # print('filtering', client.right_iris_threshold, client.right_iris_size)
                        messagebox.showinfo('Calibration', 'Next, please select the image that circled your pupil correctly')
                        current_calib_iris = False
                        current_calib_pupil = True

                elif current_calib_pupil:
                    done, calib_count = calib_eye(client, side=1, eye=right_eye, calib_count=calib_count)
                    if done:
                        calib = 3  # finish eye calibration
                        client.set_pupil_filtering(pre_left=pre_pupil_data_left, pre_right=pre_pupil_data_right,
                                             left_threshold=pre_left_pupil_threshold, right_threshold=pre_right_pupil_threshold)
                        messagebox.showinfo('Calibration', 'Right eye done!\nCalibration All Done.')

            # calibration finish
            if calib == 3:
                # detect pupil size
                left_eye = extract_eye(frame, shape, lStart, lEnd)
                right_eye = extract_eye(frame, shape, rStart, rEnd)

                # detect pupil size
                left_result, right_result = process_both_eyes(client, left_eye, right_eye)
                # print(left_size, right_size)
                # update time
                client.current_time = time.time()
                client = update_data(client, left_result, right_result)

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
