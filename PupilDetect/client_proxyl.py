from client import Client
import socket
import threading
from imutils import face_utils
import imutils
import cv2
import time
import dlib
import argparse

face_file = 'Resources/face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_file)

EYE_AR_THRESH = 0.21  # default threshold for indicate blink
EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def run_with_server(HOST = 'localhost', PORT = 8080):
    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        conn = ServerSocket.connect((HOST, PORT))
        print('Connection Established')
        while True:
            print('connection true')
            cap = cv2.VideoCapture(0)
            client = Client(ServerSocket)

            blink_start_t = time.time()
            blink_counter = 0

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

                    # get eyes
                    left_eye = client.extract_eye(frame, shape, lStart, lEnd)
                    right_eye = client.extract_eye(frame, shape, rStart, rEnd)
                    # detect pupil size
                    left_size, right_size = client.blob_process_both_eyes(left_eye, right_eye, 0, 0)

                    client.time = time.time()
                    client.left_pupil = left_size[1] if left_size is not None else 0
                    client.right_pupil = right_size[1] if right_size is not None else 0

                    # print(client.left_pupil, client.right_pupil, 'data updated!!')

                    # detecting blink
                    leftEye = shape[lStart: lEnd]
                    rightEye = shape[rStart: rEnd]
                    leftEAR = client.eye_aspect_ratio(leftEye)
                    rightEAR = client.eye_aspect_ratio(rightEye)

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
                        # print('create new process with new  -------> ')
                        comm_p = threading.Thread(target=client.send_data(blink=True))
                        # print('create new process', comm_p)
                        # print('sent blink')
                        comm_p.start()
                        comm_p.join()

                        client.blink_count = 0
                        blink_start_t = time.time()
                    elif not (client.left_pupil == 0 and client.right_pupil == 0):
                        # print('create new process with new  -------> ')
                        comm_p = threading.Thread(target=client.send_data())
                        # print('create new process', comm_p)
                        comm_p.start()
                        comm_p.join()

    except socket.error as err:
        print(err)


def run_standalone():

    cap = cv2.VideoCapture(0)
    client = Client()
    client.start_time = time.time()
    blink_counter = 0

    MAX_EAR = 0
    EAR_UNCHANGED = 0

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

            # get eyes
            left_eye = client.extract_eye(frame, shape, lStart, lEnd)
            right_eye = client.extract_eye(frame, shape, rStart, rEnd)
            # detect pupil size
            left_size, right_size = client.blob_process_both_eyes(left_eye, right_eye, 0, 0)

            client.time = time.time()
            if left_size is not None:
                client.left_pupil = left_size[1]
            if right_size is not None:
                client.right_pupil = right_size[1]

            # detecting blink
            leftEye = shape[lStart: lEnd]
            rightEye = shape[rStart: rEnd]
            leftEAR = client.eye_aspect_ratio(leftEye)
            rightEAR = client.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if EAR_UNCHANGED < 3:
                MAX_EAR = max(MAX_EAR, ear)
                if ear == MAX_EAR:
                    # print('MAX_EAR set', MAX_EAR)
                    EAR_UNCHANGED += 1
                else:
                    EAR_UNCHANGED = 0

            elif EAR_UNCHANGED == 3:
                global EYE_AR_THRESH
                EYE_AR_THRESH = MAX_EAR * 0.7
                # print('EYE_AR_THRESH SET!', EYE_AR_THRESH)
                EAR_UNCHANGED += 1 # make it to 4 so it won't run this line again

            if ear < EYE_AR_THRESH:
                blink_counter += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    client.blink_count += 1
                    print('[BLINK**]')
                # reset the eye frame counter
                blink_counter = 0

            client.show_text(frame, ear, time.time())

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='run method')
    my_parser.add_argument('--method',
                           help='server/standalone')
    my_parser.add_argument('--host',
                           help='host address to connect')

    args = my_parser.parse_args()
    print(args)
    if args.method == 'server':
        if args.host:
            run_with_server(HOST=args.host)
            print(args.host)
        else:
            run_with_server()
    elif args.method == 'standalone':
        run_standalone()
