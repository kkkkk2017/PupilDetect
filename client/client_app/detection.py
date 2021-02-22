from imutils import face_utils
import imutils
import cv2
import time
import dlib
import os.path as path
from tkinter import *
from tkinter import messagebox
import numpy as np
from Task import Task
from client import Client
import img_utility

class Program_Control:
    def __init__(self):
        self.current_calib_pupil = False
        self.current_calib_iris = True
        self.calib_count = 3
        self.calib_stage = 0
        self.task_begin = False
        # initial configuration for blink
        self.blink_set = False  # if the blink threshold set
        self.blink_counter = 0
        self.MAX_EAR = 0
        self.EAR_UNCHANGED = 0
        self.if_blink = False


face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
predictor = dlib.shape_predictor(face_file)
face_detector = dlib.get_frontal_face_detector()
storing_path = path.expanduser("~/Desktop/CSV")

global pre_pupil_data
pre_pupil_data_left = []
pre_pupil_data_right = []
pre_iris_data_left = []
pre_iris_data_right = []


EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

data_file = path.join(path.dirname(__file__), 'data.txt')
control_file = path.join(path.dirname(__file__), 'control.txt')


# -------------------- utility methods --------------------- #
def get_frame(cap):
    if cap is None: return None, None
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

def output_image(x, y, r, image, storing_path):
    image = cv2.circle(image, (int(x), int(y)), int(r), (255, 255, 255), 1)
    image = cv2.putText(image, str(x) + ' ' + str(y) + ' r= ' + str(r), (5, int(y + 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
    cv2.imwrite(storing_path, image)

# -------------------------------------------------------------------- #
def procedure(gray, compare_pupil, approx_iris,
              storing_path=None, tme=None, side=None, output_pic=False):

    iris, t = img_utility.preprocess(gray) # get x, y and threshold from the gray image
    gray = img_utility.clean_img(gray, iris[0], iris[1], approx_iris[1])

    if t is None: return None, None, None

    t -= 5
    blur = img_utility.get_binary(gray, t) # get the binary image

    cnts, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cv2.minEnclosingCircle(i) for i in cnts]    #(x, y), r
    result = img_utility.select_closest(cnts, iris[0], iris[1], 0)

    if result is None: return None, None, None

    (x, y), r = result
    # print('----[CHT RESULTS]', x, y, r)

    if compare_pupil is not None:
        _, cr = compare_pupil
        # print('similar', abs(cr-r)//1)
        if abs(cr-r)//1 <= 3:
            return gray, result, t

    if_valid, adjust_r, (adjust_x, adjust_y) = img_utility.process(blur, approx_r=r,
                                                                   iris_r=approx_iris[1], compare_pupil=compare_pupil)
    # print('if valid', if_valid)
    if not if_valid:
        if adjust_x is not None:
            x = adjust_x
        if adjust_y is not None:
            y = adjust_y
        r = adjust_r

    result = ((x, y), r)

    return gray, result, t


################### calibration ############################
def calib_iris(gray):
    r, c = gray.shape
    cols = gray.sum(axis=0)
    cols = cols / r  # average

    try:
        gradient_col = np.gradient(cols)
        gradient_col_2 = np.gradient(gradient_col)
    except:
        return None

    x = np.where(cols == min(cols))
    x = int(x[0][0])

    half_1 = gradient_col_2[:x]
    half_2 = gradient_col_2[x:]

    if len(half_1) == 0 or len(half_2) == 0: return None

    edge_1 = np.where(half_1 == min(half_1))[0]
    edge_2 = np.where(half_2 == min(half_2))[0] + x
    r = (edge_2 - edge_1) / 2
    y, _ = gray.shape
    y /= 2
    result = (x, y), r[0]

    return result

def calib_pupil(calib_pupil, client, if_left, gray, calib_count):
    if calib_pupil:
        if if_left:
            # pupil
            gray, result, thres = procedure(gray, client.approx_left_pupil, client.approx_iris)
        else:
            gray, result, thres = procedure(gray, client.approx_right_pupil, client.approx_iris)

    #find iris
    else:
        result = calib_iris(gray)
        if result is None: return False, calib_count

    if result is not None:
        (x, y), r = result
        print(result)
        image = cv2.circle(gray, (int(x), int(y)), int(r), (255, 255, 255), 1)
        if calib_pupil:
            binary = img_utility.get_binary(gray, thres)
            cv2.imshow('image', np.hstack((image, binary)))
        else:
            cv2.imshow('image', image)

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
        task_t = time.time()  # the time for send the task (per 1 sec)
        # print('video up')

        while (run != '0'):
            # print(run)
            # last_t = time.time()
            # print(last_t - client.current_time)
            run = read_control()
            #get frame
            frame, gray = get_frame(cap)
            if frame is None: continue

            # detect face
            rects = face_detector(gray, 0)
            # print('get face')
            for rect in rects:

                # detect sense organs
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # print('blink')
                # detect blink
                ear = img_utility.handle_blink(shape)
                if ear < client.EYE_AR_THRESH:
                    pc.blink_counter += 1
                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if pc.blink_counter >= EYE_AR_CONSEC_FRAMES:
                        blink += 1
                        # print('[*BLINK*] blink_count=', client.blink_count)
                    # reset the eye frame counter
                    pc.blink_counter = 0

                # print('eye')
                # get eyes
                left_eye, right_eye = img_utility.get_eye(gray, shape)
                # left_eye = img_utility.convert_gray(left_eye)  # gray image
                # right_eye = img_utility.convert_gray(right_eye)  # gray image

                # detect pupil size
                _, left_result, _ = procedure(left_eye, client.approx_left_pupil, client.approx_iris)
                _, right_result, _ = procedure(right_eye, client.approx_right_pupil, client.approx_iris)

                # update time
                client.current_time = time.time()
                client = update_data(client, left_result, right_result, blink)

                if (client.current_time - task_t) %30 == 0:
                    output_image(client.left_x, client.left_y, client.left_r, storing_path+str(client.current_time)+'_left.jpg')
                    output_image(client.right_x, client.right_y, client.right_r, storing_path+str(client.current_time)+'_right.jpg')

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
        eye = img_utility.extract_eye(frame, shape, lStart, lEnd)
        if_left = True
    elif pc.calib_stage == 2:
        if_left = False
        eye = img_utility.extract_eye(frame, shape, rStart, rEnd)
    else:
        return pc, client

    #calib iris first
    if pc.current_calib_iris:
        if (if_left and len(pre_iris_data_left) == 0) or (not if_left and len(pre_iris_data_right) == 0):
            messagebox.showinfo('Calibration', 'Calibrating iris: \nplease select the circle that are the same size as your iris.\n'
                                               'if so, press \'y\', else, press any other key.')

        done, pc.calib_count = calib_pupil(calib_pupil=False, client=client, if_left=if_left, gray=eye, calib_count=pc.calib_count)
        if done:
            messagebox.showinfo('Calibration', 'Iris done!\nThen do the same thing for the other side.')
            if if_left:
                (x, y) = client.approx_iris[0]
                r = np.mean(pre_iris_data_left)//1
                client.approx_iris = ((x, y), r)
            else:
                (x, y) = client.approx_iris[0]
                r = np.mean( (np.mean(pre_iris_data_right), client.approx_iris[1]) ) // 1
                client.approx_iris = ((x, y), r)

            # print(pre_iris_data_left, client.iris_r)
            pc.current_calib_iris = False
            pc.current_calib_pupil = True

    #calib pupil
    else:
        done, pc.calib_count = calib_pupil(calib_pupil=True, client=client, if_left=if_left, gray=eye, calib_count=pc.calib_count)

        if done:
            if if_left:
                (x, y) = client.approx_left_pupil[0]
                r = np.mean(pre_iris_data_left)//1
                client.approx_left_pupil = ((x, y), r)
            else:
                (x, y) = client.approx_right_pupil[0]
                r = np.mean(pre_iris_data_right)//1
                client.approx_right_pupil = ((x, y), r)

            pc.current_calib_iris = True
            pc.current_calib_pupil = False

            if pc.calib_stage == 2:
                # print(pre_iris_data_right, client.iris_r)

                messagebox.showinfo('Calibration', 'Right eye done!\nCalibration All Done.')
            else:
                messagebox.showinfo('Calibration', 'Left eye done!')
                pc.current_calib_pupil = False

            pc.calib_stage += 1

    return pc, client

# calibration
def run_calibration(client):
    pc = Program_Control()
    pc.calib_stage = 0 # set to 1 to skip the blink
    root = Tk()
    root.withdraw()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    client.start_time = time.time()
    ear = 0

    if pc.calib_stage == 0:
        messagebox.showinfo('Calibration',
                            'Calibrating blink: \nplease open your eye and hold still, meanwhile press \'b\' \nuntil the threshold is set')

    while True:
        frame, gray = get_frame(cap)

        if frame is None: continue
        # detect face
        rects = face_detector(gray, 0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        for rect in rects:
            # detect sense organs
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # detect blink
            ear = img_utility.handle_blink(shape)

            if client.calib_blink and key == ord('b'):
                if pc.EAR_UNCHANGED < 2:
                    pc.MAX_EAR = max(pc.MAX_EAR, ear)
                    if (ear // 0.001) == (pc.MAX_EAR // 0.001):
                        pc.EAR_UNCHANGED += 1
                    else:
                        pc.EAR_UNCHANGED = 0

                elif pc.EAR_UNCHANGED == 2:
                    client.EYE_AR_THRESH = pc.MAX_EAR * 0.75
                    pc.EAR_UNCHANGED += 1  # make it to 5 so it won't run this line again
                    client.calib_blink = False  # calibration for blink
                    messagebox.showinfo('Calibration', 'Blink SET! {} Please then blink 3 times'.format(client.EYE_AR_THRESH))
                    pc.blink_set = True
                    time.sleep(1)

            if client.blink_count == 3:
                if pc.calib_stage == 0:
                    pc.calib_stage = 1
                    messagebox.showinfo('Calibration',
                                    'Please select the image that circled your iris correctly by press \'y\', '
                                    'otherwise, anyother key.\nWhen you ready, press Any Key')

            if ear < client.EYE_AR_THRESH and pc.blink_set:
                pc.blink_counter += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if pc.blink_counter >= EYE_AR_CONSEC_FRAMES:
                    client.blink_count += 1
                    pc.if_blink = True
                else:
                    pc.if_blink = False
                # reset the eye frame counter
                pc.blink_counter = 0

            pc, client = eye_calibration_stage(gray, shape, pc, client)

            # calibration finish
            if pc.calib_stage == 3:
                # detect pupil size
                if not pc.if_blink:
                    left_eye, right_eye = img_utility.get_eye(gray, shape)
                    # left_eye = img_utility.convert_gray(left_eye) # gray image
                    # right_eye = img_utility.convert_gray(right_eye) # gray image

                    # detect pupil size
                    _, left_result, _ = procedure(left_eye, client.approx_left_pupil, client.approx_iris)
                    _, right_result, _ = procedure(right_eye, client.approx_right_pupil, client.approx_iris)

                    # update time
                    client.current_time = time.time()
                    client = update_data(client, left_result, right_result, None)

        show_text(client, frame, ear, time.time())
        cv2.imshow('Frame', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.reset_start_time()
    client.reset_data()
    return client
