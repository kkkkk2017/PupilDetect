import cv2
import time
import os
import img_utility
import argparse
import numpy as np
import imutils

# participant = 'light_eye'
(lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#histogram approx center + activate contours
def image_process(rects, frame, t, output_file):
    t = str(t)
    for rect in rects:
        shape = img_utility.get_shape(frame, rect)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[lStart:lEnd]]))
        left_eye = frame[y:(y + h), x:(x + w)]
        left_eye = imutils.resize(left_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[rStart:rEnd]]))
        right_eye = frame[y:(y + h), x:(x + w)]
        right_eye = imutils.resize(right_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

        cv2.imwrite(output_file + "_RGB_left_" + t + ".jpg", left_eye)
        cv2.imwrite(output_file + "_RGB_right_" + t + ".jpg", right_eye)

        #hsv
        left_hsv = cv2.cvtColor(left_eye, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(right_eye, cv2.COLOR_BGR2HSV)

        if left_hsv is not None:
            cv2.imwrite(output_file+"_HSV_left_"+t+".jpg", left_hsv)

        if right_hsv is not None:
            cv2.imwrite(output_file+"_HSV_right_"+t+".jpg", right_hsv)



# ************************* ********** . NOTE . ************ *************** ******* #
# version:
#  bug from last version: some peaks are too close - inappropriate peaks
# 1. find contours -> select one
# 2. use histogram to adjust the radius
#       crop the area to half, then get the peak on each side
#       the distance between the peaks

def do_video(filename, reading_dir, storing_dir):
    print('START')

    video = os.path.join(reading_dir, filename)
    cap = cv2.VideoCapture(video)

    file = filename[:-4]
    print(file)

    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            print('frame is None')
            break
        # time.sleep(0.05)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = img_utility.face_detector(gray, 0)
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        image_process(rects, frame, t, os.path.join(storing_dir, file))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('DONE')


if __name__ == '__main__':
    # do_image()
    # for i in range(1, 7):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('parent_dir', type=str)
    # parser.add_argument('output', type=bool)
    # parser.add_argument('output_file', type=str)
    # args = parser.parse_args()
    # args = vars(args)

    parent_dir = 'D:/PupilDilateProject/test_video'
    parent_dir = os.path.realpath(parent_dir)

    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            second_dir = os.path.join(parent_dir, dir)
            if second_dir.endswith('_photos'): continue

            result_dir = os.path.join(parent_dir, second_dir+'_photos')
            #create folder
            print(result_dir)
            if os.path.isdir(result_dir):
                print('existed, pass')
                continue #avoid overwritten the previous results

            os.mkdir(result_dir)
            print(result_dir, 'created')

            for filename in os.listdir(second_dir):
                print(filename)
                if filename.endswith(".mp4"):
                    do_video(filename, second_dir, result_dir)

    # do_video(args.file_name, args.output, args.output_file)
    #'1back'
