import cv2
import time
import os
import numpy as np
import img_utility

approx_iris = (165.5, 39.5, 66.5)
compare_pupil = (100, 60.5, 40)

light_storing_path = os.path.expanduser('~/Desktop/light_images/')
dark_storing_path = os.path.expanduser('~/Desktop/dark_images/')
storing_path = dark_storing_path


def procedure(gray, storing_path, tme, side, compare_pupil, output_pic):
    # gray = img_utility.convert_gray(eye_image) # gray image

    iris, t = img_utility.preprocess(gray, side) # get x, y and threshold from the gray image
    gray = img_utility.clean_img(gray, iris[0], iris[1], approx_iris[2])

    if t is None: return None, None

    blur = img_utility.get_binary(gray, t-20, side, tme, output_pic, storing_path) # get the binary image

    cnts, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cv2.minEnclosingCircle(i) for i in cnts]    #(x, y), r

    result = img_utility.select_closest(cnts, iris[0], iris[1], 0)


    if result is None: return None, None
    (x, y), r = result
    # print('----[CHT RESULTS]', x, y, r)

    if compare_pupil is not None:
        _, cr = compare_pupil
        # print('similar', abs(cr-r)//1)
        if abs(cr-r)//1 <= 6:
            # print('CONTOURS')
            print('-------- [FINAL RESULT]', x, y, r, '\n')
            if output_pic:

                gray = cv2.circle(gray, (int(x), int(y)), int(r), (255, 255, 255), 1)

                gray = cv2.putText(gray, str(x) + ' ' + str(y) + ' r= ' + str(r), (5, int(y + 15)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 25)
                cv2.imwrite(storing_path, gray)
            return (result, t)


    if_valid, adjust_r, (adjust_x, adjust_y) = img_utility.process(blur, iris_r=approx_iris[2], compare_pupil=None,
                                                                   time=tme, side=side, output_pic=output_pic, storing_path=storing_path)
    # print('if valid', if_valid)
    if not if_valid:
        if adjust_x is not None:
            x = adjust_x
        if adjust_y is not None:
            y = adjust_y
        r = adjust_r

        # print('*[HIST ADJUST] adjust radius =', r, 'adjust x = ', x)

    print('-------- [FINAL RESULT]', x, y, r, '\n')
    if output_pic:

        gray = cv2.circle(gray, (int(x), int(y)), int(r), (255, 255, 255), 1)
        gray = cv2.putText(gray, str(x) + ' ' + str(y) + ' r= ' + str(r), (5, int(y + 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.imwrite(storing_path, gray)
    result = ((x, y), r)

    return (result, t)


#histogram approx center + activate contours
def image_process(rects, gray, frame, s, csv_path, compare_pupil, output_pic):

    for rect in rects:
        shape = img_utility.get_shape(gray, rect)

        left_eye, right_eye = img_utility.get_eye(gray, shape)
        tme = time.time() - s

        (left_pupil, t_left) = procedure(left_eye, tme=str(tme), side='left', compare_pupil=compare_pupil,
                                         output_pic=output_pic, storing_path=storing_path + str(tme)+'left_eye.jpg')
        (right_pupil, t_right) = procedure(right_eye, tme=str(tme), side='right', compare_pupil=compare_pupil,
                                           output_pic=output_pic, storing_path=storing_path + str(tme)+'right_eye.jpg')
        # print('left pupil', left_pupil)
        # print('right pupil', right_pupil)
        if left_pupil is None and right_pupil is None: continue

        with open(csv_path, 'a') as f:
            f.write(str(time.time()) + ',')
            if left_pupil is not None:
                f.write(str(left_pupil[0][0]) + ',')
                f.write(str(left_pupil[0][1]) + ',')
                f.write(str(left_pupil[1]) + ',')
            else:
                f.write(str(0) + ',')

            f.write(str(t_left) + ',')
            if right_pupil is not None:
                f.write(str(right_pupil[0][0]) + ',')
                f.write(str(right_pupil[0][1]) + ',')
                f.write(str(right_pupil[1]) + ',')
            else:
                f.write(str(0) + ',')

            # f.write(str(t_right) + ',')
            f.write('\n')
        f.close()


# ************************* ********** . NOTE . ************ *************** ******* #
# version:
#  bug from last version: some peaks are too close - inappropriate peaks
# 1. find contours -> select one
# 2. use histogram to adjust the radius
#       crop the area to half, then get the peak on each side
#       the distance between the peaks

def do_video(color, output_pic):

    csv_name = 'pupil_data_indoor_1.csv'
    video_name = 'dark_video_sun.mp4'

    if color == 'l':
        csv_name = 'light_' + csv_name
        video_name = 'light_video_1.mp4'
        global storing_path
        storing_path = light_storing_path

    csv_path = os.path.expanduser('~/Desktop/test_video/'+csv_name)
    with open(csv_path, 'w') as f:
        f.write('time' + ',')
        f.write('left_x' + ',')
        f.write('left_y' + ',')
        f.write('left_r' + ',')
        f.write('left_t' + ',')
        f.write('right_x' + ',')
        f.write('right_y' + ',')
        f.write('right_r' + ',')
        f.write('right_t' + ',')
        f.write('\n')
    f.close()

    video = os.path.expanduser('~/Desktop/test_video/'+video_name)
    cap = cv2.VideoCapture(video)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    print('[OPEN CAP]', cap.isOpened())

    s = time.time()
    compare_pupil = (133, 32), 17 #25 light pupil
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            print('frame is None')
            break
        # time.sleep(0.05)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = img_utility.face_detector(gray, 0)
        image_process(rects, gray, frame, s, csv_path, compare_pupil, output_pic)
        # print('[Process Time] ', time.time() - s, '\n')
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # do_image()
    # for i in range(1, 7):
    do_video('d', False)