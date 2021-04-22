import cv2
import time
import os
import img_utility
import argparse

# approx_iris = (165.5, 39.5, 66.5)
# compare_pupil = (100, 60.5, 40)

light_storing_path = os.path.expanduser('~/Desktop/light_images/')
dark_storing_path = os.path.expanduser('~/Desktop/dark_images/')
storing_path = dark_storing_path


def procedure(gray, storing_path, tme, side, compare_pupil, output_pic):
    # gray = img_utility.convert_gray(eye_image) # gray image

    iris, t = img_utility.preprocess(gray, side) # get x, y and threshold from the gray image
    gray = img_utility.clean_img(gray, iris[0], iris[1], approx_iris[2])
    if t is None: return None, None

    # cv2.imwrite(storing_path + str(time) + side + 'pupil.jpg', pupil_range)

    blur = img_utility.get_binary(gray, t-20, side, tme, output_pic, storing_path) # get the binary image

    cnts, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cv2.minEnclosingCircle(i) for i in cnts]    #(x, y), r

    result = img_utility.select_closest(cnts, compare_pupil[0][0], compare_pupil[0][1], compare_pupil[1])
    if result is None: return None, None
    (x, y), r = result
    # print('----[CHT RESULTS]', x, y, r)

    if compare_pupil is not None:
        _, cr = compare_pupil
        # print('similar', abs(cr-r)//1)
        if r >= (cr * 0.7) and r <= (cr*1.3):
            # print('-------- CONTOURS [FINAL RESULT]', x, y, r, '\n')
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
        r = adjust_r

        # if adjust_x is not None:
        #     x = adjust_x
        # if adjust_y is not None:
        #     y = adjust_y

        # print('*[HIST ADJUST] adjust radius =', r, 'adjust x = ', x)
    if r < (compare_pupil[1] * 0.7) or r > (compare_pupil[1]*1.3): return None, None

    # if r >= compare_pupil[1] * 1.4 or r <= compare_pupil[1] * 0.6: return None, None

    # print('-------- [FINAL RESULT]', x, y, r, '\n')
    if output_pic:
        gray = cv2.circle(gray, (int(x), int(y)), int(r), (255, 255, 255), 1)
        gray = cv2.putText(gray, str(x) + ' ' + str(y) + ' r= ' + str(r), (5, int(y + 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.imwrite(storing_path, gray)
    result = ((x, y), r)

    return (result, t)


#histogram approx center + activate contours
def image_process(rects, gray, tme, csv_path, compare_pupil, output_pic):

    for rect in rects:
        shape = img_utility.get_shape(gray, rect)

        left_eye, right_eye = img_utility.get_eye(gray, shape)

        (left_pupil, t_left) = procedure(left_eye, tme=str(tme), side='left', compare_pupil=compare_pupil,
                                         output_pic=output_pic, storing_path=storing_path + str(tme)+'left_eye.jpg')

        (right_pupil, t_right) = procedure(right_eye, tme=str(tme), side='right', compare_pupil=compare_pupil,
                                           output_pic=output_pic, storing_path=storing_path + str(tme)+'right_eye.jpg')
        # print('left pupil', left_pupil)
        # print('right pupil', right_pupil)
        if left_pupil is None and right_pupil is None: continue

        with open(csv_path, 'a') as f:
            f.write(str(tme) + ',')
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

            f.write(str(t_right) + ',')
            f.write('\n')
        f.close()


# ************************* ********** . NOTE . ************ *************** ******* #
# version:
#  bug from last version: some peaks are too close - inappropriate peaks
# 1. find contours -> select one
# 2. use histogram to adjust the radius
#       crop the area to half, then get the peak on each side
#       the distance between the peaks

def do_video(file_name, output_pic, output_file):

    csv_name = output_file + '.csv'
    # video_name = 'dark_video_sun.mp4'

    global approx_iris
    approx_iris = (165.5, 39.5, 68)
    global compare_pupil
    compare_pupil = ((133, 52),18)

    if output_file.__contains__('light'):
        global storing_path
        storing_path = light_storing_path
        approx_iris = (165.5, 39.5, 68)
        compare_pupil = ((133, 52), 20)

    video_name = file_name+'.mp4'
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

    # print('[OPEN CAP]', cap.isOpened())

    s = time.time()
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            print('frame is None')
            break
        # time.sleep(0.05)

        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = img_utility.face_detector(gray, 0)
        image_process(rects, gray, t, csv_path, compare_pupil, output_pic)
        # print('[Process Time] ', time.time() - s, '\n')
        # cv2.imshow('frame', frame)
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
    for i in ['1back', '2back', '3back', 'video_task']:
        do_video(i, False, 'light_'+i)