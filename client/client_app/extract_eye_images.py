import cv2
import time
import os
import img_utility
import argparse
import numpy as np
import imutils

participant = 'light_eye'
storing_path = os.path.expanduser('~/Desktop/test_eye_images/' + participant + '/')
(lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#histogram approx center + activate contours
def image_process(rects, frame, count, output_file):

    for rect in rects:
        shape = img_utility.get_shape(frame, rect)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[lStart:lEnd]]))
        left_eye = frame[y:(y + h), x:(x + w)]
        left_eye = imutils.resize(left_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[rStart:rEnd]]))
        right_eye = frame[y:(y + h), x:(x + w)]
        right_eye = imutils.resize(right_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

        cv2.imwrite(output_file + "/RGB/left_" + str(count) + ".jpg", left_eye)
        cv2.imwrite(output_file + "/RGB/right_" + str(count) + ".jpg", right_eye)

        #hsv
        left_hsv = cv2.cvtColor(left_eye, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(right_eye, cv2.COLOR_BGR2HSV)

        if left_hsv is not None:
            h, s, v = cv2.split(left_hsv)
            cv2.imwrite(output_file+"/HSV_H/left_"+str(count)+".jpg", h)
            cv2.imwrite(output_file+"/HSV_S/left_"+str(count)+".jpg", s)
            cv2.imwrite(output_file+"/GRAY/left_"+str(count)+".jpg", v)
            cv2.imwrite(output_file+"/HSV/left_"+str(count)+".jpg", left_hsv)

            h = img_utility.convert_gray(h)
            s = img_utility.convert_gray(s)
            # h[h > 110] = 0
            v = img_utility.convert_gray(v)
            hsv = cv2.merge((h, s, v))
            cv2.imwrite(output_file + "/HSV/left_cvt_" + str(count) + ".jpg", hsv)
            cv2.imwrite(output_file+"/HSV_H/left_cvt_"+str(count)+".jpg", img_utility.convert_gray(h))
            cv2.imwrite(output_file+"/HSV_S/left_cvt_"+str(count)+".jpg", img_utility.convert_gray(s))
            cv2.imwrite(output_file+"/GRAY/left_cvt_"+str(count)+".jpg", img_utility.convert_gray(v))
            # left_eye = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


        if right_hsv is not None:
            h, s, v = cv2.split(right_hsv)
            cv2.imwrite(output_file+"/HSV/right_"+str(count)+".jpg", right_hsv)
            cv2.imwrite(output_file+"/HSV_H/right_"+str(count)+".jpg", h)
            cv2.imwrite(output_file+"/HSV_S/right_"+str(count)+".jpg", s)
            cv2.imwrite(output_file+"/GRAY/right_"+str(count)+".jpg", v)

            h = img_utility.convert_gray(h)
            s = img_utility.convert_gray(s)
            # h[h > 110] = 0
            v = img_utility.convert_gray(v)
            hsv = cv2.merge((h, s, v))
            cv2.imwrite(output_file + "/HSV/right_cvt_" + str(count) + ".jpg", hsv)
            cv2.imwrite(output_file+"/HSV_H/right_cvt_"+str(count)+".jpg", h)
            cv2.imwrite(output_file+"/HSV_S/right_cvt_"+str(count)+".jpg", s)
            cv2.imwrite(output_file+"/GRAY/right_cvt_"+str(count)+".jpg", v)

            # right_eye = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ************************* ********** . NOTE . ************ *************** ******* #
# version:
#  bug from last version: some peaks are too close - inappropriate peaks
# 1. find contours -> select one
# 2. use histogram to adjust the radius
#       crop the area to half, then get the peak on each side
#       the distance between the peaks

def do_video(input_file, output_file):

    video_name = input_file+'.mp4'

    video = os.path.expanduser('~/Desktop/'+video_name)
    cap = cv2.VideoCapture(video)

    count = 0

    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            print('frame is None')
            break
        # time.sleep(0.05)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = img_utility.face_detector(gray, 0)
        image_process(rects, frame, count, output_file)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # do_image()
    # for i in range(1, 7):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('file_name', type=str)
    # parser.add_argument('output', type=bool)
    # parser.add_argument('output_file', type=str)
    # args = parser.parse_args()
    # args = vars(args)
    # do_video(args.file_name, args.output, args.output_file)
    #'1back'
    do_video(participant, storing_path)
