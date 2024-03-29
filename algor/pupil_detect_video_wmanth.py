import cv2
import os
from os import listdir, path
from os.path import join
import numpy as np
from scipy.spatial import distance
import eyeball_detect as ed
import tools
import argparse
import img_utility
import time
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist

RGB = []
avg_iris = 61
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# select result by euclidean distance and count pixels of ROI
# roi here is eyeball
def select_by_pd(results, iris_r, point_2):
    if len(results) == 0:
        return None, None

    result_set = [ (calculate_weighted_distance(i, point_2), i)
                    for i in results]
    result_set = list(sorted(result_set, key=lambda x: x[0]))

    valid_result = []
    for i in result_set:
        if i[0] < 20:
            valid_result.append(i)

    if len(valid_result) == 0:
        return None, None

    # log.write('\ndistance set, avg pixel = ' + str(result_set))
    # print('- sorted result set', result_set)

    first = np.array(valid_result[0]).flatten()

    final_result = first[1]

    log.write('\n** final result = ')
    log.write(str(final_result) + ' | ratio= ' + str(np.around(final_result[2] / iris_r, decimals=2)))
    log.write('\nweighted similarity = ' + str(first[0]))
    # print('final result', final_result, '| ratio=', np.around(final_result[2] / iris_r, decimals=2))

    return first


def approach_hough_cnt(roi, original_width, calib_result):
    # output = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    # print('\n-------[APPROACH CNT]-----------------------------------')
    height, width = roi.shape
    iris_r = min(height, width) / 2

    # log.write('\nROI size (w,h) ' + str(width) + ' ' + str(height))
    # print('ROI size ' + str(width) + ' ' + str(height))

    pre_x, pre_y = tools.get_preliminary_center(roi)
    dis_to_center = distance.euclidean((pre_x, pre_y), (width / 2, height / 2))
    # print('dis to cenetr =', dis_to_center)

    if np.around(dis_to_center, decimals=0) > 15:
        pre_x = int(width / 2)
        pre_y = int(height / 2)
    # log.write('\npreliminary center ' + str(pre_x) + ',' + str(pre_y))
    # print('preliminary center', pre_x, pre_y)

    # cv2.imwrite(storing_path + 'ap1_roi_' + file, roi)

    if width < original_width / 3:
        # log.write('\n' + str(width))
        log.write('\n** ROI is too small')
        # print('\n** ROI is too small')
        return None, None

    t = tools.get_threshold(roi)
    # log.write('\nthreshold = ' + str(t))
    # print('threshold', t)
    # print('threshold = ' + str(max_t) + "<-max|min->" + str(min_t))

    _, binary = cv2.threshold(roi, t, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (7, 7))
    # cv2.imwrite(storing_path + 'binary_' + file, binary)

    ## trial 2: contours
    contour_circles, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = [cv2.minEnclosingCircle(i) for i in contour_circles]
    contour_circles = [[a, b, c] for (a, b), c in contour_circles]

    if contour_circles is None:
        log.write('\nresult is None')
        # print('** Result is None')
        return None, None

    ##  select the results
    circle_results = np.around(contour_circles)
    # log.write('\ncircle reuslt: \n')
    for i in circle_results:
        log.write(str(np.around(i[2] / iris_r, decimals=2)) + '\t')

    results_new = []
    for i in circle_results:
        if tools.if_radius_valid(i[2], iris_r):
            results_new.append(i)
    circle_results = results_new

    if len(circle_results) == 0: return None, None

    (sim, final_result) = select_by_pd(circle_results, iris_r, calib_result)

    # hist_result = histogram_on_roi(pre_x, pre_x, iris_r, binary)
    # print('histogram on result 1', hist_result)

    # print('final result', final_result)
    if final_result is None: return None, None

    # output = cv2.circle(roi, (pre_x, pre_y), 2, (0, 0, 255), 3)
    # output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), 2, (0, 255, 0), 2)
    # output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), int(final_result[2]),
    #                     (0, 255, 0), 2)
    # cv2.imwrite(storing_path + 'ap1_pupil_' + file, output)

    return (sim, final_result)


def set_center(pixels, point):
    left_i, right_i = point - 1, point + 1 #ignore the point
    current_value = pixels[point]
    while left_i > 0 and right_i < len(pixels):
        a = pixels[left_i]
        b = pixels[right_i]
        if a >= current_value and b < current_value:
            point = left_i
            current_value = a
        elif a < current_value and b >= current_value:
            point = right_i
            current_value = b
        else:
            break

        left_i -= 1
        right_i += 1

    return point


CENTER_PIXEL_THRESH = 245

def expanding(array, start, start_value, iris_r, PIXEL_THRESHOLD=15):
    a = 0
    b = 0
    for idx, value in enumerate(reversed(array[:start])):
        a = idx
        # print(idx, value)
        if abs(value - start_value) >= PIXEL_THRESHOLD:
            break
    # print('\n')
    for idx, value in enumerate(array[start:]):
        b = idx
        # print(idx, value)
        if abs(value - start_value) >= PIXEL_THRESHOLD:
            break

    # print('a, b radius', a, b)
    r = (a + b) / 2
    x = b + start - r
    # print('radius x|y, r', x, r)
    if tools.if_radius_valid(r, iris_r):
        return x, r
    else:
        return x, None


# roi - eyeball
# result: coordinate only
def approach_expand(roi, result, calib_result):
    # print('\n-------[APPROACH EXPAND FROM CENTER]-------------------------')

    height, width = roi.shape
    # print(height, width)

    iris_r = min(height, width) / 2
    r = 0

    if result is not None and not (result[0] == 0 and result[1] == 0)\
            and not (result[1] >= height or result[0] >= width):
        result = np.around(result, decimals=0)
        pre_x, pre_y = result
    else:
        sum_row = roi.sum(axis=0) #sum of each rows, y
        sum_col = roi.sum(axis=1) #sum of each columns, x
        pre_y = np.where(sum_col == max(sum_col))[0][0] #where the maximum col
        pre_x = np.where(sum_row == max(sum_row))[0][0] #where the maximum row
        # print('pre x|y from global maxi', pre_x, pre_y)

    # print('pre result', pre_x, pre_y)
    # log.write('\npre coordinate = ' + str(pre_x) + ' ' + str(pre_y))

    pre_x = int(pre_x)
    pre_y = int(pre_y)

    select_row = roi[pre_y]  # east, west
    select_col = roi[:, pre_x]  # north, south

    #check by pixel values
    x_value = int(select_row[pre_x])
    y_value = int(select_col[pre_y])

    # if the pixel value is too low, which means the center is incorrect
    # if x_value < CENTER_PIXEL_THRESH or y_value < CENTER_PIXEL_THRESH:
    #     # find center again
    #     new_x, new_y, times = adjust_center(pre_x, pre_y, roi, times, calib_result)
    #     if times == 0:
    #         return pre_x, pre_y, r
    #     else:
    #         return approach_expand(roi, (new_x, new_y), times, calib_result)

    x, hor_r = expanding(select_row, pre_x, x_value, iris_r) # east, west
    # print('x, hor r', x, hor_r)
    # log.write('\nx, hor r: ' + str(x) + '|' + str(hor_r))
    y, ver_r = expanding(select_col, pre_y, y_value, iris_r) #north, south
    # print('y, ver r', y, ver_r)
    # log.write('\ny, ver r: ' + str(y) + '|' + str(ver_r))

    if hor_r is not None and ver_r is not None:
        r = np.mean( [hor_r, ver_r])
    elif hor_r is not None:
        r = hor_r
    elif ver_r is not None:
        r = ver_r

    # log.write('\n** final result = ')
    # log.write(str(x) + ',' + str(y) + ' | ratio= ' + str(
    #     np.around(r / iris_r, decimals=2)) + ' iris r = ' + str(iris_r))

    # print('final result', x, y, r, np.around(r / iris_r, decimals=2), iris_r)
    sim = calculate_weighted_distance((x,y,r), calib_result)
    return sim, (x,y,r)

# create mask
# x, y, r = result
# h, w = img.shape
# circle_image = np.full(img.shape[:2], 255, np.uint8)
# circle_image = cv2.circle(circle_image, (int(x), int(y)), int(r), 0, thickness=-1)
# masked_data = cv2.bitwise_not(v_channel, mask=circle_image)
# cv2.imwrite(storing_path + 'ap2_masked_' +file, masked_data)

#weighted euclidean distance
def calculate_weighted_distance(point_1, point_2):
    return np.around(dist.euclidean(point_1, point_2, (0.2, 0.45, 0.35)), decimals=2)

def input_calibration(dir, path):
    d = dir.split('_')
    d = d[0]
    left_x, left_y, left_r = [], [], []
    right_x, right_y, right_r = [], [], []

    # with open(os.path.join(path, d, 'calibration.txt'), 'r') as f:
    with open('D:/PupilDilateProject/calibration.txt', 'r') as f:
        for line in f.readlines():
            values = line.split(',')
            if values[0] == 'left':
                left_x.append(int(values[1]))
                left_y.append(int(values[2]))
                left_r.append(int(values[3]))

            elif values[0] == 'right':
                right_x.append(int(values[1]))
                right_y.append(int(values[2]))
                right_r.append(int(values[3]))

    return (np.around(np.mean(left_x) ,decimals=0),np.around(np.mean(left_y) ,decimals=0),
            np.around(np.mean(left_r) ,decimals=0)),(np.around(np.mean(right_x) ,decimals=0),
                                                     np.around(np.mean(right_y) ,decimals=0),np.around(np.mean(right_r) ,decimals=0))

def select_channel(h, s, v):
    hp = np.average(h)
    sp = np.average(s)
    vp = np.average(v)

    select = sorted([hp, sp, vp])[1]
    if select == hp:
        print('select h')
        return h, select
    if select == sp:
        print('select s')
        return s, select
    if select == vp:
        print('select v')
        return v, select

def process_single_image(rgb, log, calib_reuslt):
    original_width, original_height = rgb.shape[:2]

    # log.write('\n finding iris')
    # print('[finding iris]')
    crop_area, roi_gray, roi_rgb = ed.extract_eyeball_ROI(rgb)

    h, w = roi_gray.shape[:2]
    iris_r = min(h, w) / 2
    if roi_gray is None:
        # log.write('\nROI is None')
        print('** ROI is None')
        return None

    v_channel = tools.image_denoise(roi_gray)
    v_channel_inv = cv2.bitwise_not(v_channel)

    v_channel_inv = cv2.equalizeHist(v_channel_inv)
    v_channel_inv = cv2.medianBlur(v_channel_inv, 5)

    x, y, r, sim = 0, 0, 0, 0
    current_calib = (calib_reuslt[0] - crop_area[0], calib_reuslt[1], calib_reuslt[2])

    sim1, result_1 = approach_hough_cnt(v_channel, original_width, current_calib)
    sim2, result_2 = approach_hough_cnt(v_channel_inv, original_width, current_calib)

    # log.write('\nresult 1 ' + str(result_1))
    # log.write(' | similarity = ' + str(sim1))
    # log.write('\nresult 2 ' + str(result_2))
    # log.write(' | similarity = ' + str(sim2))

    # select from first two approach first
    if result_1 is not None:
        x, y, r = result_1
        sim = sim1

    if result_2 is not None:
        x2, y2, r2 = result_2
        sim = sim2

        if x != x2 and y != y2 and r != r2:
            pixel_1, _ = tools.check_roi_pixels(x, y, r, roi_gray)
            pixel_2, _ = tools.check_roi_pixels(x2, y2, r2, roi_gray)
            # print('result 1 = ', pixel_1, ' vs result 2 =', pixel_2)

            if pixel_1 > pixel_2:
                x, y, r = result_2
                sim = sim2

    # select with the third
    sim3, (x3, y3, r3) = approach_expand(v_channel_inv, (x, y), current_calib)

    # log.write('\nresult 3 ' + str(x3) + ',' + str(y3) + ',' + str(r3) + ' compare with result ' + str(x) + ',' + str(
    #     y) + ',' + str(r))

    pixel_1, _ = tools.check_roi_pixels(x, y, r, roi_gray)
    pixel_3, _ = tools.check_roi_pixels(x3, y3, r3, roi_gray)

    # log.write('\nmean pixel values' + str(pixel_1) + ',' + str(pixel_3))
    if pixel_1 > pixel_3 or (x == 0 and y == 0 and r == 0) and r3 is not None:
        x, y, r, sim = x3, y3, r3, sim3

    x += crop_area[0]  # crop for getting the eyeball

    if r == 0: x, y, sim = 0, 0, 0

    # print('[FINAL RESULT]', x, y, r, 'with similarity = ', sim, '\n')
    ratio = np.around(r / iris_r, decimals=2)

    if ratio >= 0.55 or ratio <= 0.3: return None

    # log.write('\n[FINAL RESULT] ' + str(x) + ',' + str(y) + ',' + str(r)
    #           + ' ' + str(ratio))

    if r == 0: return None

    return (x,y, r, sim, ratio)


def write_result(csv, step, ts, result, side, d2):
    x,y,r,sim,ratio = result

    # csv.write(step + ',')
    csv.write(ts + ',')
    csv.write(str(x) + ',')
    csv.write(str(y) + ',')
    csv.write(str(r) + ',')
    csv.write(str(ratio) + ',')
    csv.write(str(sim) + ',')
    csv.write(side + ',')
    csv.write(str(d2) + ',')
    csv.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str) #test/eva
    args = parser.parse_args()
    args = vars(args)
    mode = args['mode']

    global storing_path
    #uncomment to change way of storing files

    # RGB = [f for f in listdir(input_path) if isfile(join(input_path, f))]

    #another way to access files
    reading_path = []

    if mode == 'eva': #evaluation
        # parent_dir = 'D:/PupilDilateProject/test_video'
        # parent_dir = os.path.realpath(parent_dir)
        # for subdir, dirs, files in os.walk(parent_dir):
        #     for dir in dirs:
        #         if not dir.__contains__('_'):
        #             reading_path.append(dir)
        parent_dir = 'D:/eve_dataset/eve_dataset'
        parent_dir = os.path.realpath(parent_dir)
        for root, dirs, files in os.walk(parent_dir):
            if dirs.__len__() == 0:
                reading_path.append(root)

    elif mode == 'test':
        condition = 'test'  # only works when detected area is smaller than actual, not works on conditon 1
        # parent_dir = path.expanduser('~/Desktop/test_video/')
        parent_dir = 'D:/eve_dataset/eve_dataset'
        reading_path.append(parent_dir)

    for dir in reading_path:
        #dir = train01_photos
        print('dir in reading path', dir)
        if mode == 'eva':
            # storing_path = os.path.join(parent_dir, dir+'_video_result')
            # print('storing path', storing_path)
            #
            # if os.path.isdir(storing_path): continue #avoid overwrite the previous results.
            #
            # os.mkdir(storing_path)
            append = True
            for f in listdir(dir):
                if f.endswith('.csv'):
                    append = False
                    break

            if append:
                storing_path = dir
                print(storing_path)
            else:
                continue

            #input calibration data
            calib_left, calib_right = input_calibration(dir, parent_dir)

        else:
            left_x, left_y, left_r = [], [], []
            right_x, right_y, right_r = [], [], []

            with open(os.path.join(parent_dir, 'calibration.txt'), 'r') as f:
                for line in f.readlines():
                    values = line.split(',')
                    if values[0] == 'left':
                        left_x.append(int(values[1]))
                        left_y.append(int(values[2]))
                        left_r.append(int(values[3]))

                    elif values[0] == 'right':
                        right_x.append(int(values[1]))
                        right_y.append(int(values[2]))
                        right_r.append(int(values[3]))

            calib_left = (np.around(np.mean(left_x), decimals=0), np.around(np.mean(left_y), decimals=0),
                    np.around(np.mean(left_r), decimals=0))
            calib_right = (np.around(np.mean(right_x), decimals=0),np.around(np.mean(right_y), decimals=0),
                                                              np.around(np.mean(right_r), decimals=0))

        calib_d2 = np.around(dist.euclidean(calib_left[:2], calib_right[:2]), decimals=2)
        print('calib_d2', calib_d2)

        second_dir = os.path.join(parent_dir, dir)
        log = open(os.path.join(storing_path, "opencv_video_log.txt"), "w")
        csv = open(os.path.join(storing_path, "opencv_video_result.csv"), 'w')

        # csv.write('step' + ',')
        csv.write('ts' + ',')
        csv.write('x' + ',')
        csv.write('y' + ',')
        csv.write('pupil' + ',')
        csv.write('ratio' + ',')
        csv.write('sim' + ',')
        csv.write('side' + ',')
        csv.write('d2' + ',')
        csv.write('\n')

        print('second_dir', second_dir)

        for file in listdir(second_dir): #all video in the folder

            if not file.endswith('webcam_c_face.mp4'): continue

            print('file in second dir', file)

            digs = file[:-4].split('_')
            step = digs[1]

            log.write('\n\nstep ' + step + " --------------------")
            print('step ' + step + " --------------------")

            video = os.path.join(second_dir, file)
            cap = cv2.VideoCapture(video)
            start_t = time.time()

            while (cap.isOpened()):
                try:
                    _, frame = cap.read()
                    if frame is None:
                        print('frame is None')
                        break
                    # time.sleep(0.05)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = img_utility.face_detector(gray, 0)
                    t = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t = str(t)

                    for rect in rects:
                        shape = img_utility.get_shape(frame, rect)

                        ear = img_utility.handle_blink(shape)
                        if ear < 0.21: continue

                        (x, y, w, h) = cv2.boundingRect(np.array([shape[lStart:lEnd]]))
                        left_eye = frame[y:(y + h), x:(x + w)]
                        left_eye = imutils.resize(left_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

                        (x, y, w, h) = cv2.boundingRect(np.array([shape[rStart:rEnd]]))
                        right_eye = frame[y:(y + h), x:(x + w)]
                        right_eye = imutils.resize(right_eye, width=100 * 3, height=75 * 3, inter=cv2.INTER_CUBIC)

                        left_result = process_single_image(left_eye, log, calib_left)
                        right_result = process_single_image(right_eye, log, calib_right)

                        d_eyes = -1
                        if left_result is not None and right_result is not None:
                            lx, ly = left_result[:2]
                            rx, ry = right_result[:2]
                            d_eyes = np.around(dist.euclidean((lx, ly), (rx,ry)), decimals=2)
                            # print('two eyes d', d_eyes)
                            # log.write('\ntwo eyes interval ' + str(d_eyes))
                            # if abs(d_eyes - calib_d2) <= 20:
                            #     if left_result[-1] < right_result[-2]:
                            #         right_result = None
                            #     else:
                            #         left_result = None

                        if left_result is not None:
                            write_result(csv, step, t, left_result, 'L', d_eyes)
                            # output = left_eye
                            # x,y,r = left_result[:3]
                            # output = cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), thickness=2)
                            # output = cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), thickness=2)
                            # cv2.imwrite(os.path.join(storing_path, step+'_'+t+'_left.png'), output)

                        if right_result is not None:
                            write_result(csv, step, t, right_result, 'R', d_eyes)
                            # output = right_eye
                            # x,y,r = right_result[:3]
                            # output = cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), thickness=2)
                            # output = cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), thickness=2)
                            # cv2.imwrite(os.path.join(storing_path, step+'_'+t+'_right.png'), output)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    continue

                cap.release()
                cv2.destroyAllWindows()
                log.write('\n[FINISH] Process time = ' + str(time.time() - start_t))
                print('[FINISH] Process time = ', time.time() - start_t)


        #
        csv.close()
        log.close()

