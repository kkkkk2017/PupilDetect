from imutils import face_utils
import imutils
import cv2
import time
import dlib
import os.path as path
from tkinter import *
from tkinter import messagebox
from scipy.spatial import distance as dist
import numpy as np
from Task import Task
from client import Client

class Program_Control:
    def __init__(self):
        self.current_calib_pupil = False
        self.current_calib_iris = True
        self.calib_count = 3
        self.calib_stage = 0
        self.task_begin = False


face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
predictor = dlib.shape_predictor(face_file)
face_detector = dlib.get_frontal_face_detector()

global pre_pupil_data
pre_pupil_data_left = []
pre_pupil_data_right = []
pre_iris_data_left = []
pre_iris_data_right = []


EYE_AR_THRESH = 0.21  # default threshold for indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

data_file = path.join(path.dirname(__file__), 'data.txt')
control_file = path.join(path.dirname(__file__), 'control.txt')

# ---------------------- get eye methods ----------------------------- #
def get_eyes(frame, shape):
    left_eye = extract_eye(frame, shape, lStart, lEnd)
    right_eye = extract_eye(frame, shape, rStart, rEnd)
    return left_eye, right_eye

def extract_eye(frame, shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:(y + h), x:(x + w)]
    roi = imutils.resize(roi, width=100, height=75, inter=cv2.INTER_CUBIC)
    return roi

def convert_gray(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(image)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (5, 5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (5, 5))
    return gray


def get_binary(gray, t):
    if t is None or gray is None: return None

    _, thres = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    blur = cv2.medianBlur(open, 3)
    return blur


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # vertical
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# -------------------- utility methods --------------------- #
def get_frame(cap):
    _, frame = cap.read()
    frame = imutils.resize(frame, width=1000, height=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def show_text(client, frame, ear, time):
    cv2.putText(frame, "Blinks: {}".format(client.blink_count), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.putText(frame, 'Left Pupil: {:.5f}'.format(client.current_left_pupil_r), (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, 'Right Pupil: {:.5f}'.format(client.current_right_pupil_r), (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.putText(frame, 'Time {:2f}'.format(time - client.start_time), (10, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


# --------------------- main detection algorithm -------------------------- #
def get_peak(half_ary, reverse=False):
    if len(half_ary) == 0:
        # print('input ary is empty, [ERROR FIND PEAKS]')
        return 0

    peak_value = max(half_ary)
    peak = np.where(half_ary == peak_value)[0]
    if reverse:
        return peak[-1]
    else:
        return peak[0]

# check if the radius is valid
def valid_radius(pupil_r, iris_r):
    ratio = pupil_r/iris_r
    # print('ratio', ratio, pupil_r)
    if ratio >= 0.2 and ratio <= 0.6: return True
    return False

def col_gray(gray):
    cols = []

    r, c = gray.shape
    # [0-100]
    for a in range(c):
        sum = 0
        for b in range(r):
            sum += gray[b][a]
        cols.append(sum / c)

    cols = cols[10:-10]
    return cols

def row_gray(gray):
    rows = []
    r, _ = gray.shape
    for v in gray:
        rows.append(np.sum(v) / r)
    rows = rows[5:-5]
    return rows

# step 0: detect the approximate x, y and threshold
# step 1: detect the quality of the blur image, adjust the most likely radius
def show_histogram(gray, step, area_x=None, area_y=None, area_r=None, compare_pupil=None, approx_r=6, iris_r=20):
    rows = row_gray(gray)
    cols = col_gray(gray)

    try:
        gradient_row = np.gradient(rows)
        gradient_col = np.gradient(cols)

        gradient_col_2 = np.gradient(gradient_col)
        gradient_row_2 = np.gradient(gradient_row)

    except:
        return (None, None),  None

    x = np.where(cols == min(cols))
    x = int(x[0].mean())
    y = np.where(gradient_row_2 == max(gradient_row_2))
    y = int(y[0].mean())

    if step == 0:
        return (x, y), min(cols)+15

    elif step == 1:
        if area_x is not None and area_y is not None and area_r is not None:
            if area_x >= 10: area_x -= 10
            if area_y >= 5: area_y -= 5

            peaks_col = [get_peak(gradient_col_2[:area_x], reverse=True),  get_peak(gradient_col_2[area_x:])+area_x]
            peaks_row = [get_peak(gradient_row_2[:area_y], reverse=True),  get_peak(gradient_row_2[area_y:])+area_y]

            col_r = abs(peaks_col[1] - peaks_col[0])/2
            row_r = abs(peaks_row[1] - peaks_row[0])/2

            col_valid = valid_radius(col_r, iris_r)
            row_valid = valid_radius(row_r, iris_r)

            if col_valid and row_valid and abs(col_r - row_r) <= 3: True, None, (None, None) # no need to adjust the radius

            if not col_valid and not row_valid: return None, None, (None, None) # cannot adjust the radius

            elif row_valid: return False, row_r, (None, None)
            elif col_valid: return False, col_r, (
                    ((peaks_col[1] + peaks_col[0])/2)+10, ((peaks_row[1]+peaks_row[0])/2)+5 )
            else:
                if approx_r is not None:
                    if abs(approx_r - row_r) <= abs(approx_r - col_r):
                        return False, row_r, (None, None)
                    else: return False, col_r, (
                    ((peaks_col[1] + peaks_col[0])/2)+10, ((peaks_row[1]+peaks_row[0])/2)+5 )

            return False, col_r, (None, None)

        return True, None, (None, None)
    return None, None, (None, None)


def filt_results(cnts):
    cnts = [cv2.minEnclosingCircle(i) for i in cnts]     #(x, y), r
    results = []
    for cnt in cnts:
        if cnt[-1] >= 20:
            continue
        else:
            results.append(cnt)

    if len(results) == 0:
        return None

    results = sorted(results, key=lambda x: x[-1], reverse=True)
    return results[0]

def procedure(eye_image, compare_pupil, approx_r=6, iris_r=20):
    gray = convert_gray(eye_image) # gray image
    (x, y), t = show_histogram(gray, 0) # get x, y and threshold from the gray image
    if t is None: return gray, None, None

    blur = get_binary(gray, t) # get the binary image
    # circles = find_circle(blur) # circle detections

    cnts, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = filt_results(cnts)

    if result is None: return gray, None, None
    (x, y), r = result
    # print('----[CNT RESULTS]', x, y, r)

    if_valid, adjust_r, (adjust_x, adjust_y) = show_histogram(blur, 1, int(x//1), int(y//1),
                                                              r, compare_pupil, approx_r, iris_r)
    if if_valid is None: return gray, None, None

    if not if_valid:
        if adjust_x is not None:
            if x == adjust_x:
                # print('*** adjutst x and contour x are the same, no need to adjust **')
                pass
            x = adjust_x
            y = np.mean((adjust_y, y))
            y = int(y)
        r = adjust_r

        # print('*[HIST ADJUST] adjust radius =', r, 'adjust x = ', x)

    # print('-------- [FINAL RESULT]', x, y, r, '\n')
    result = ((x, y), r)
    return gray, result, t


################### calibration ############################
def handle_blink(shape):
    # detecting blink
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return np.around(ear, decimals=2)


def calib_pupil(calib_pupil, client, if_left, eye, calib_count):
    if calib_pupil:
        if if_left:
            # pupil
            gray, result, thres = procedure(eye, client.prev_left_pupil, client.left_approx_r, client.iris_r)
        else:
            gray, result, thres = procedure(eye, client.prev_right_pupil, client.right_approx_r, client.iris_r)
        # print('selected threshold', thres)
        binary = get_binary(eye, thres)
    # print('detection result', pupil)
    else:
        binary = None
        gray = convert_gray(eye)  # gray image
        cols = col_gray(gray)

        try:
            gradient_col = np.gradient(cols)
            gradient_col_2 = np.gradient(gradient_col)
        except:
            return False, calib_count

        x = np.where(cols == min(cols))
        x = int(x[0][0])

        half_1 = gradient_col_2[:x]
        half_2 = gradient_col_2[x:]

        if len(half_1) == 0 or len(half_2) == 0: return False, calib_count

        edge_1 = np.where(half_1 == min(half_1))[0]
        edge_2 = np.where(half_2 == min(half_2))[0]+x
        r = (edge_2 - edge_1)/2
        y, _ = gray.shape
        y /= 2
        result = (x+10, y), r

    if result is not None:
        (x, y), r = result
        image = cv2.circle(gray, (int(x), int(y)), int(r), (255, 255, 255), 1)
        if binary is None:
            cv2.imshow('image', image)
        else:
            cv2.imshow('image', np.hstack((image, binary)))

        if cv2.waitKey(20000) & 0xFF == ord('y'):
            cv2.destroyWindow('image')
            calib_count -= 1
            if calib_count != 0:
                messagebox.showinfo('Sample count Left', '{} left'.format(calib_count))

            if if_left:
                if calib_pupil:
                    pre_pupil_data_left.append(result[-1]) #only radius
                else:
                    pre_iris_data_left.append(result[-1])
                if calib_count == 0:
                    return True, 3
            else:
                if calib_pupil:
                    pre_pupil_data_right.append(result[-1])
                else:
                    pre_iris_data_right.append(result[-1])
                if calib_count == 0:
                    return True, 3
        else:
            cv2.destroyWindow('image')

    return False, calib_count


def read_control():
    with open(control_file, 'r') as f:
        lines = f.readline(1)
        f.close()
    return lines[0]


def store_data(task):
    # print(task)
    with open(data_file, 'a') as f:
        if task:
            for i in task.to_List():
                f.write(str(i) + ',')
            f.write('\n')
    f.close()


def update_data(client, left_result, right_result, blink_count):
    if left_result is not None:
        (x, y), r = left_result
        client.current_left_pupil_x = x
        client.current_left_pupil_y = y
        client.current_left_pupil_r = r

    if right_result is not None:
        (x, y), r = right_result
        client.current_right_pupil_x = x
        client.current_right_pupil_y = y
        client.current_right_pupil_r = r

    if blink_count is not None:
        client.blink_count = blink_count
    return client


##################### main program #########################
def run(client=None):
    pc = Program_Control()
    if client is None:
        client = Client()

    with open(data_file, 'w') as f:
        pass

    run = read_control()

    # last_update = time.time()
    blink = 0

    while True:
        if run == '0':
            print('run is 0')
            break

        cap = cv2.VideoCapture(0)

        blink_t = time.time()  # the time for send the blink (per 10 sec)
        # task_t = time.time()  # the time for send the task (per 1 sec)
        blink_counter = 0
        # print('video up')

        while (run != '0'):
            # print(run)
            # last_t = time.time()
            # print(last_t - client.current_time)
            run = read_control()
            #get frame
            frame, gray = get_frame(cap)

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
                        blink += 1
                        # print('[*BLINK*] blink_count=', client.blink_count)
                    # reset the eye frame counter
                    blink_counter = 0

                # print('eye')
                # get eyes
                left_eye, right_eye = get_eyes(gray, shape)

                # detect pupil size
                _, left_result, _ = procedure(left_eye, compare_pupil=None)
                _, right_result, _ = procedure(right_eye, compare_pupil=None)

                # update time
                client.current_time = time.time()
                client = update_data(client, left_result, right_result, blink)

                if run == 'n':
                    if pc.task_begin:
                        blink = 0
                        pc.task_begin = False

                if run == 'a':
                    if not pc.task_begin:
                        pc.task_begin = True
                        blink_t = time.time()  # the time for sending the blink (per 10 sec)
                        blink = 0

                    if client.current_time - blink_t >= 10:
                        t = Task(time=client.current_time,
                                 left_pupil_r=client.current_left_pupil_r, left_pupil_x=client.current_left_pupil_x,
                                 left_pupil_y=client.current_left_pupil_y,
                                 right_pupil_r=client.current_right_pupil_r, right_pupil_x=client.current_right_pupil_x,
                                 right_pupil_y=client.current_right_pupil_y,
                                 blink_count=blink)
                        blink = 0
                        store_data(t)
                        client.reset_data()
                        blink_t = time.time()
                        # task_t = time.time()
                    else:
                        # elif client.current_time - task_t >= 0.001:
                        if not (client.current_left_pupil_r == 0 and client.current_right_pupil_r == 0):
                            t = Task(time=client.current_time,
                                     left_pupil_r=client.current_left_pupil_r, left_pupil_x=client.current_left_pupil_x,
                                     left_pupil_y=client.current_left_pupil_y,
                                     right_pupil_r=client.current_right_pupil_r, right_pupil_x=client.current_right_pupil_x,
                                     right_pupil_y=client.current_right_pupil_y,
                                     blink_count=np.nan)
                            store_data(t)
                            client.reset_data()
                            # task_t = time.time()

    cap.release()
    cv2.destroyAllWindows()


def eye_calibration_stage(frame, shape, pc, client):
    if pc.calib_stage == 1:
        eye = extract_eye(frame, shape, lStart, lEnd)
        if_left = True
    elif pc.calib_stage == 2:
        if_left = False
        eye = extract_eye(frame, shape, rStart, rEnd)
    else:
        return pc, client

    if pc.current_calib_iris:
        if (if_left and len(pre_iris_data_left) == 0) or (not if_left and len(pre_iris_data_right) == 0):
            messagebox.showinfo('Calibration', 'Calibrating iris: \nplease select the circle that are the same size as your iris.\n'
                                               'if so, press \'y\', else, press anyother key.')
        done, pc.calib_count = calib_pupil(calib_pupil=False, client=client, if_left=if_left, eye=eye, calib_count=pc.calib_count)
        if done:
            messagebox.showinfo('Calibration', 'Iris done!\nThen do the same thing for pupil.')
            client.iris_r = np.average(pre_iris_data_left)

            # print(pre_iris_data_left, client.iris_r)
            pc.current_calib_iris = False
            pc.current_calib_pupil = True
    else:
        done, pc.calib_count = calib_pupil(calib_pupil=True, client=client, if_left=if_left, eye=eye, calib_count=pc.calib_count)

        if done:
            pc.current_calib_iris = True
            pc.current_calib_pupil = False

            if pc.calib_stage == 2:
                client.iris_r = np.average([client.iris_r, np.average(pre_iris_data_right)])
                # print(pre_iris_data_right, client.iris_r)

                client.left_approx_r = np.average(pre_pupil_data_left)
                client.right_approx_r = np.average(pre_pupil_data_right)
                messagebox.showinfo('Calibration', 'Right eye done!\nCalibration All Done.')
            else:
                messagebox.showinfo('Calibration', 'Left eye done!')
                pc.current_calib_pupil = False

            pc.calib_stage += 1

    return pc, client


def run_standalone(client):
    pc = Program_Control()
    pc.calib_stage = 0 # set to 1 to skip the blink
    root = Tk()
    root.withdraw()
    messagebox.showinfo('Calibration', 'Please open your eyes and stay for 1 second.\nWhen you ready, press Any Key')
    cap = cv2.VideoCapture(0)

    client.start_time = time.time()

    #initial configuration for blink
    blink_set = False #if the blink threshold set
    blink_counter = 0
    MAX_EAR = 0
    EAR_UNCHANGED = 0
    if_blink = False

    while True:
        frame, gray = get_frame(cap)

        # detect face
        rects = face_detector(gray, 0)
        for rect in rects:
            # detect sense organs
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # detect blink
            ear = handle_blink(shape)

            key = cv2.waitKey(1) & 0xFF
            if client.calib_blink and key == ord('b'):
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
                    client.calib_blink = False  # calibration for blink
                    messagebox.showinfo('Calibration', 'Blink SET! {} Please then blink 3 times'.format(EYE_AR_THRESH))
                    time.sleep(1)

            if client.blink_count == 3:
                if pc.calib_stage == 0:
                    pc.calib_stage = 1
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
                    if_blink = True
                else:
                    if_blink = False
                # reset the eye frame counter
                blink_counter = 0

            pc, client = eye_calibration_stage(gray, shape, pc, client)

            # calibration finish
            if pc.calib_stage == 3:
                # detect pupil size
                if not if_blink:
                    left_eye, right_eye = get_eyes(gray, shape)

                    # detect pupil size
                    _, left_result, _ = procedure(left_eye, client.prev_left_pupil)
                    _, right_result, _ = procedure(right_eye, client.prev_right_pupil)

                    # update time
                    client.current_time = time.time()
                    client = update_data(client, left_result, right_result, None)

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
