import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import matplotlib.patches as mpatches

plt.style.use('seaborn-whitegrid')

face_cascade = cv2.CascadeClassifier('Casacades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Casacades/haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

left_pupil_data = []
right_pupil_data = []
pupil_avg = []


# load image and resize
def load_image(file_name):
    image = cv2.imread(file_name)
    # scale_percent = 60  # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # image = cv2.resize(image, (dim))

    return image


# detect face
def detect_face(image, classifier):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(img_gray, 1.3, 5, )
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # filter the small objects
    if len(faces) > 1:
        biggest = (0, 0, 0, 0)
        for i in faces:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(faces) == 1:
        biggest = faces
    else:
        return None
    for (x, y, w, h) in biggest:
        face_frame = image[y:y + h, x:x + w]
    return face_frame


# detect eyes
def detect_eyes(image, classifier):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(img_gray, 1.3, 5)  # detect eyes
    width = np.size(image, 1)  # get face frame width
    height = np.size(image, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if y > height / 2:  # ignore the bottom half of the face
            pass
        eye_center = x + w / 2  # get the eye center
        if eye_center < width * 0.5:
            left_eye = image[y:y + h, x:x + w]
        else:
            right_eye = image[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(eye):
    height, width = eye.shape[:2]
    eyebrow_h = int(height / 4)  # eyebrows always take ~25% of the image
    eye = eye[eyebrow_h:height, 0:width]
    return eye


def blob_process(image, detector, threshold):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(image, None, iterations=2)
    # erode = image
    dilate = cv2.dilate(erode, None, iterations=5)

    blur = cv2.medianBlur(dilate, 5)
    output = np.hstack((gray_frame, image, erode, dilate, blur))
    cv2.imshow('Compare', output)
    keypoints = detector.detect(image)
    return keypoints


def get_keypoints(image, detector, threshold):
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    image = cv2.erode(image, None, iterations=2)
    image = cv2.dilate(image, None, iterations=5)
    image = cv2.medianBlur(image, 5)
    keypoints = detector.detect(image)
    return keypoints


def blob_process_both_eyes(left_eye, right_eye, detector, start_thres):
    if start_thres < 0 or start_thres > 255:
        return None

    left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

    for thres in range(20, 160):  # 0 - 255
        left_keypoints = get_keypoints(left_gray, detector, thres)
        right_keypoints = get_keypoints(right_gray, detector, thres)
        if len(left_keypoints) > 0 and len(right_keypoints) > 0:
            return thres, left_keypoints, right_keypoints
    return None

def main_image():
    frame = load_image('portrait.jpg')
    face = detect_face(frame, face_cascade)
    le, re = detect_eyes(face, eye_cascade)
    le = cut_eyebrows(le)
    re = cut_eyebrows(re)


    le_keypoints = None
    re_keypoints = None

    le_keypoints = blob_process(le, detector, 40)
    re_keypoints = blob_process(re, detector, 40)
    thres, le_keypoints, re_keypoints = blob_process_both_eyes(le, re, detector, 40)

    print('thres=', thres)
    if le_keypoints or re_keypoints:
        left_d = le_keypoints[0].size
        right_d = re_keypoints[0].size

        cv2.drawKeypoints(le, le_keypoints, le, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(re, re_keypoints, re, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Pupil Detector', frame)

    c = cv2.waitKey()
    cv2.destroyAllWindows()


def nothing(x):
    pass


def main_with_trackbar():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Pupil Detector')
    cv2.createTrackbar('threshold', 'Pupil Detector', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)
        if face_frame is not None:
            face_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            eyes = detect_eyes(face_frame, eye_cascade)  # left eye & right eye
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'Pupil Detector')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, detector, threshold)
                    cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    _, face_image = cv2.threshold(face_gray, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('Pupil Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    fig, ax = plt.subplots()
    legend_element = [mpatches.Patch(color='blue', label='Left pupil', linestyle='-', linewidth=0.5),
                      mpatches.Patch(color='green', label='Right pupil', linestyle='-', linewidth=0.5),
                      mpatches.Patch(color='red', label='Mean', linestyle='--', linewidth=0.5)]
    ax.legend(handles=legend_element, loc='upper_left')
    ax.set_ylim(bottom=0, top=15)
    cap = cv2.VideoCapture(0)
    thres = 30

    while True:
        _, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)

        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)  # left eye & right eye
            if len(eyes) == 2:
                left_eye, right_eye = eyes

                le_pupil = None
                re_pupil = None

                if (left_eye is not None) and (right_eye is not None):
                    left_eye = cut_eyebrows(left_eye)
                    right_eye = cut_eyebrows(right_eye)

                    value = blob_process_both_eyes(left_eye, right_eye, detector, thres)
                    if value is not None:
                        thres, le_pupil, re_pupil = value
                        print('thres=', thres)

                        lpd = np.around(le_pupil[0].size, decimals=5)
                        rpd = np.around(re_pupil[0].size, decimals=5)
                        avg = np.around((lpd + rpd) / 2, decimals=5)
                        if lpd < 10 and rpd < 10:
                            left_pupil_data.append(lpd)
                            right_pupil_data.append(rpd)
                            pupil_avg.append(avg)

                            ax.plot([i for i in range(len(left_pupil_data))], left_pupil_data, '-b')
                            ax.plot([i for i in range(len(right_pupil_data))], right_pupil_data, '-g')
                            ax.plot([i for i in range(len(pupil_avg))], pupil_avg, '--r',)
                            fig.canvas.draw()

                    cv2.drawKeypoints(left_eye, le_pupil, left_eye, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.drawKeypoints(right_eye, re_pupil, right_eye, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        cv2.imshow('Pupil Detector', frame)
        fig.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
# main_image()
# main_with_trackbar()