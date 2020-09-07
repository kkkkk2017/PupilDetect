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

EYE_AR_THRESH = 0.21  # indicate blink
# EYE_AR_CONSEC_FRAMES = 3  # for the number of consecutive frames the eye must be below the threshold
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def run_with_server(HOST = 'localhost', PORT = 8080):
    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        conn = ServerSocket.connect((HOST, PORT))

        while True:
            cap = cv2.VideoCapture(0)
            client = Client(ServerSocket)

            blink_start_t = time.time()
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

                    if left_size and right_size:
                        left_pupil, right_pupil = left_size[1], right_size[1]
                        client.time = time.time()
                        client.left_pupil = left_pupil
                        client.right_pupil = right_pupil
                        print(left_pupil, right_pupil, 'data updated!!')

                        # detecting blink
                        leftEye = shape[lStart: lEnd]
                        rightEye = shape[rStart: rEnd]
                        leftEAR = client.eye_aspect_ratio(leftEye)
                        rightEAR = client.eye_aspect_ratio(rightEye)

                        ear = (leftEAR + rightEAR) / 2.0
                        print(ear)

                        if ear < EYE_AR_THRESH:
                            client.blink_count += 1
                            print('[*BLINK*] blink_count=', client.blink_count)

                        if time.time() - blink_start_t >= 10:
                            # print('create new process with new  -------> ')
                            comm_p = threading.Thread(target=client.send_data(blink=True))
                            # print('create new process', comm_p)
                            print('sent blink')
                            comm_p.start()
                            comm_p.join()

                            client.blink_count = 0
                            blink_start_t = time.time()
                        else:
                            # print('create new process with new  -------> ')
                            comm_p = threading.Thread(target=client.send_data())
                            # print('create new process', comm_p)
                            comm_p.start()
                            comm_p.join()

    except socket.error as err:
        print(err)
    finally:
        exit(0)

def run_standalone():

    cap = cv2.VideoCapture(0)
    client = Client()
    client.start_time = time.time()

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

            if left_size and right_size:
                left_pupil, right_pupil = left_size[1], right_size[1]
                client.time = time.time()
                client.left_pupil = left_pupil
                client.right_pupil = right_pupil

                # detecting blink
                leftEye = shape[lStart: lEnd]
                rightEye = shape[rStart: rEnd]
                leftEAR = client.eye_aspect_ratio(leftEye)
                rightEAR = client.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                if ear < EYE_AR_THRESH:
                    client.blink_count += 1

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
    if args.method == 'server':
        if args.host:
            run_with_server(HOST=args.host)
        else:
            run_with_server()
    elif args.method == 'standalone':
        run_standalone()
