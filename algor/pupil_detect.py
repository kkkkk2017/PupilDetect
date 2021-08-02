import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
from scipy.spatial import distance
import eyeball_detect as ed
import tools
import argparse

condition = 'eyeballs'  # only works when detected area is smaller than actual, not works on conditon 1
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'

RGB = []
avg_iris = 61

# select result by euclidean distance and count pixels of ROI
# roi here is eyeball
def select_by_pd(eyeball, results, pre_x, pre_y, iris_r):
    result_set = [tools.check_roi_pixels(int(i[0]), int(i[1]), int(i[2]), eyeball) for i in results]
    result_set = list(sorted(result_set, key=lambda x: (x[0], distance.euclidean((x[1][0], x[1][1]), (pre_x, pre_y)))))

    f.write('\ndistance set, avg pixel = ' + str(result_set))
    print('- sorted result set', result_set)

    first = np.array(result_set[0]).flatten()
    if first[0] == 255: return None

    final_result = first[1]
    f.write('\n** final result = ')
    f.write(str(final_result) + ' | ratio= ' + str(np.around(final_result[2] / iris_r, decimals=2)))
    print('final result', final_result, '| ratio=', np.around(final_result[2] / iris_r, decimals=2))

    return final_result


def approach_hough_cnt(roi, file, original_width):
    output = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    print('\n-------[APPROACH HGH + CNT]-----------------------------------')
    height, width = roi.shape
    iris_r = min(height, width) / 2

    f.write('\nROI size (w,h) ' + str(width) + ' ' + str(height))
    print('ROI size ' + str(width) + ' ' + str(height))

    pre_x, pre_y = tools.get_preliminary_center(roi)
    dis_to_center = distance.euclidean((pre_x, pre_y), (width / 2, height / 2))
    print('dis to cenetr =', dis_to_center)

    if np.around(dis_to_center, decimals=0) > 15:
        pre_x = int(width / 2)
        pre_y = int(height / 2)
    f.write('\npreliminary center ' + str(pre_x) + ',' + str(pre_y))
    print('preliminary center', pre_x, pre_y)

    # cv2.imwrite(storing_path + 'ap1_roi_' + file, roi)

    if width < original_width / 3:
        f.write('\n' + str(width))
        f.write('\n** ROI is too small')
        print('\n** ROI is too small')
        return None

    max_t, min_t = tools.get_threshold(roi)
    f.write('\nthreshold = ' + str(max_t) + "<-max|min->" + str(min_t))
    print('threshold = ' + str(max_t) + "<-max|min->" + str(min_t))

    _, binary = cv2.threshold(roi, min_t - 15, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (7, 7))
    cv2.imwrite(storing_path + 'ap1_binary_' + file, binary)

    ## trial 1: hough circle
    # gray_circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=30, minRadius=0, maxRadius=100)

    binary_circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=20, param2=10, minRadius=0, maxRadius=100)

    ## trial 2: contours
    contour_circles, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = [cv2.minEnclosingCircle(i) for i in contour_circles]
    contour_circles = [[a, b, c] for (a, b), c in contour_circles]

    if contour_circles is not None and binary_circles is not None:
        circle_results = np.append(contour_circles, binary_circles[0], axis=0)
    else:
        circle_results = contour_circles if contour_circles is not None else binary_circles[0]
        # circle_results = binary_circles

    if circle_results is None:
        f.write('\nresult is None')
        print('** Result is None')
        return None

    ##  select the results
    circle_results = np.around(circle_results)
    f.write('\ncircle reuslt: \n')
    for i in circle_results:
        f.write(str(np.around(i[2] / iris_r, decimals=2)) + '\t')

    results_new = []
    for i in circle_results:
        if tools.if_radius_valid(i[2], iris_r):
            results_new.append(i)
    circle_results = results_new

    if len(circle_results) == 0: return None

    final_result = select_by_pd(roi, circle_results, pre_x, pre_x, iris_r)

    hist_result = histogram_on_roi(pre_x, pre_x, iris_r, binary)
    print('histogram on result 1', hist_result)

    print('final result', final_result)
    if final_result is None: return None

    output = cv2.circle(output, (pre_x, pre_y), 2, (0, 0, 255), 3)
    output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), 2, (0, 255, 0), 2)
    output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), int(final_result[2]),
                        (0, 255, 0), 2)
    cv2.imwrite(storing_path + 'ap1_pupil_' + file, output)

    return final_result


def histogram_on_roi(x, y, r, img):
    x = int(x)
    y = int(y)
    r = int(r)
    roi = img[y - r:y + r, x - r:x + r]

    if roi is None or len(roi) == 0: return 0, 0, 0

    # roi here should be pupil roi
    _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    if binary is None: return 0, 0, 0

    # print('------- CHECK ROI BINARY IMAGE (' + str(x) + ',' + str(y) + ',' + str(r) + ') ------------------ ')
    # print('input result', x, y, r)

    h, w = binary.shape
    iris_r = min(h,w)/2
    # [0-100]
    cols = binary.sum(axis=0)
    rows = binary.sum(axis=1)

    cols, gradient_col_2, _, _, _ = tools.get_roi_attributes(cols, h, False)
    rows, gradient_row_2, _, _, _ = tools.get_roi_attributes(rows, w, False)

    middle_row = int(np.where(rows == min(rows))[0][0])
    middle_col = int(np.where(cols == min(cols))[0][0])

    half = gradient_col_2[:middle_col]
    sec_half = gradient_col_2[middle_col + 1:]
    peaks_col = [tools.get_peak(half, if_back=True), tools.get_peak(sec_half) + middle_col]

    half = gradient_row_2[:middle_row]
    sec_half = gradient_row_2[middle_row + 1:]
    peaks_row = [tools.get_peak(half, if_back=True), tools.get_peak(sec_half) + middle_row]

    assum_x = (peaks_col[1] + peaks_col[0]) / 2
    assum_y = (peaks_row[1] + peaks_row[0]) / 2

    col_r = abs(peaks_col[1] - peaks_col[0]) / 2
    row_r = abs(peaks_row[1] - peaks_row[0]) / 2

    assum_x += (x - r)
    assum_y += (y - r)

    assum_x = max(assum_x, 0)
    assum_y = max(assum_y, 0)

    print('assume (x,y)', assum_x, assum_y, col_r, row_r)
    f.write('\nassume (x,y, col r, row r) ' + str([assum_x, assum_y, col_r, row_r]))

    return assum_x, assum_y, max(col_r, row_r)


def appraoch_inv(eyeball, result):
    # roi here is eyeball

    output = cv2.cvtColor(eyeball, cv2.COLOR_GRAY2RGB)
    h, w = eyeball.shape[:2]
    iris_r = min(h, w)/2

    print('\n-------[APPROACH INV]-----------------------------------')
    max_t, min_t = tools.get_threshold(eyeball)
    f.write('\nthreshold = ' + str(max_t) + "<-max|min->" + str(min_t))
    print('threshold = ' + str(max_t) + "<-max|min->" + str(min_t))

    _, binary = cv2.threshold(eyeball, max_t + 15, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (7, 7))
    cv2.imwrite(storing_path + 'ap2_binary_' + file, binary)

    binary_circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=20, param2=10, minRadius=0, maxRadius=100)

    ## trial 2: contours
    contour_circles, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = [cv2.minEnclosingCircle(i) for i in contour_circles]
    contour_circles = [(a, b, c) for (a, b), c in contour_circles]

    if contour_circles is not None and binary_circles is not None:
        circle_results = np.append(contour_circles, binary_circles[0], axis=0)
    else:
        circle_results = contour_circles if contour_circles is not None else binary_circles[0]
        # circle_results = binary_circles

    if circle_results is None:
        f.write('\nresult is None')
        print('** Result is None')
        return None

    ##  select the results
    if result is not None:
        circle_results = np.append(circle_results, [result], axis=0)
    circle_results = np.around(circle_results)

    f.write('\ncircle reuslt: \n')
    for i in circle_results:
        f.write(str(np.around(i[2] / iris_r, decimals=2)) + '\t')

    if len(circle_results) == 0: return None

    distance_set = [(tools.check_roi_pixels(i[0], i[1], i[2], eyeball,
                                            mode=1, recur=True))
                    for i in circle_results]

    results_new = []
    for i in distance_set:
        if tools.if_radius_valid(i[1][2], iris_r) and i[0] != 0 :
            results_new.append(i)
    if len(results_new) == 0: return None

    distance_set = results_new

    print('distance set', distance_set)
    print('iris r', iris_r)
    f.write('\ndistance set ' + str(distance_set))

    # distance_set = [(value[0], histogram_on_roi(value[1][0], value[1][1], value[1][2], eyeball))
    #                 for value in distance_set]
    #
    # results_new = []
    # for i in distance_set:
    #     if tools.if_radius_valid(i[1][2], iris_r) and i[0] != 0:
    #         results_new.append(i)
    #
    # if len(results_new) == 0: return None
    #
    # distance_set = results_new
    # print('update distance set', distance_set)

    distance_set = list(sorted(distance_set, key=lambda x: x[0], reverse=True))

    f.write('\ndistance set, avg pixel = ' + str(distance_set))
    print('ap 2 distance set', distance_set)
    final_result = distance_set[0]

    if final_result[0] == 0: return None

    final_result = final_result[1]

    f.write('\n** final result = ')
    f.write(str(final_result[0]) + ',' + str(final_result[1]) + ' | ratio= ' + str(
        np.around(final_result[2] / iris_r, decimals=2)))
    print('final result', final_result[0], final_result[1], '| ratio=', np.around(final_result[2] / iris_r, decimals=2))

    if result is not None:
        output = cv2.circle(output, (int(result[0]), int(result[1])), 2, (0, 0, 255), 3)

    output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), 2, (0, 255, 0), 2)
    output = cv2.circle(output, (int(final_result[0]), int(final_result[1])), int(final_result[2]),
                        (0, 255, 0), 2)
    cv2.imwrite(storing_path + 'ap2_pupil_' + file, output)

    return final_result


CENTER_PIXEL_THRESH = 245
def adjust_center(origin_x, origin_y, image, times):
    print('*** adjust center * ',  times)
    f.write('\n *** adjust center * ' + str(times))
    if times == 0: return origin_x, origin_y, 0

    height, width = image.shape

    if origin_x == width-1 or origin_y == height-1: return origin_x, origin_y, times-1

    # check which side's neighbour has lower value, start from the higher side
    h = image[origin_y][origin_x - 1] + image[origin_y][origin_x + 1]
    v = image[:, origin_x][origin_y - 1] + image[:, origin_x][origin_y + 1]

    # if horizontal axis has higher value, which means this side is more like tune correct,
    # adjust y first
    if h >= v:
        vertical = image[:, origin_x]  # north, south
        vertical[origin_y] = 0 # do not consider original coordinate
        y = np.where(vertical == max(vertical))[0][0]
        horizon = image[y, :]  # east, west
        horizon[origin_x] = 0
        x = np.where(horizon == max(horizon))[0][0]

    else:
        horizon = image[origin_y, :]  # east, west
        horizon[origin_x] = 0
        x = np.where(horizon == max(horizon))[0][0]
        vertical = image[:, x]  # north, south
        vertical[origin_y] = 0
        y = np.where(vertical == max(vertical))[0][0]

    f.write('adjust x,y = ' + str(x) + ',' + str(y))
    print('adjust center =', x, y)
    return x, y, times-1


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

    print('a, b radius', a, b)
    r = (a + b) / 2
    x = b + start - r
    print('radius x|y, r', x, r)
    if tools.if_radius_valid(r, iris_r):
        return x, r
    else:
        return x, None


# roi - eyeball
# result: coordinate only
def approach_expand(roi, result, times):
    print('\n-------[APPROACH EXPAND FROM CENTER]-------------------------')

    height, width = roi.shape
    print(height, width)

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
        print('pre x|y from global maxi', pre_x, pre_y)

    print('pre result', pre_x, pre_y)
    f.write('\npre coordinate = ' + str(pre_x) + ' ' + str(pre_y))

    pre_x = int(pre_x)
    pre_y = int(pre_y)

    print(roi.shape[0], pre_y)
    select_row = roi[pre_y]  # east, west
    select_col = roi[:, pre_x]  # north, south

    #check by pixel values
    x_value = int(select_row[pre_x])
    y_value = int(select_col[pre_y])

    # if the pixel value is too low, which means the center is incorrect
    if x_value < CENTER_PIXEL_THRESH or y_value < CENTER_PIXEL_THRESH:
        # find center again
        new_x, new_y, times = adjust_center(pre_x, pre_y, roi, times)
        if times == 0:
            return pre_x, pre_y, r
        else:
            return approach_expand(roi, (new_x, new_y), times)

    x, hor_r = expanding(select_row, pre_x, x_value, iris_r) # east, west
    print('x, hor r', x, hor_r)
    f.write('\nx, hor r: ' + str(x) + '|' + str(hor_r))
    y, ver_r = expanding(select_col, pre_y, y_value, iris_r) #north, south
    print('y, ver r', y, ver_r)
    f.write('\ny, ver r: ' + str(y) + '|' + str(ver_r))

    if hor_r is not None and ver_r is not None:
        r = np.mean( [hor_r, ver_r])
        print('mean ', r)
        # pixel_1, _ = tools.check_roi_pixels(x, y, hor_r, roi, mode=1)
        # pixel_2, _ = tools.check_roi_pixels(x, y, ver_r, roi, mode=1)
        #
        # if pixel_1 <= pixel_2:
        #     r = hor_r
        # else:
        #     r = ver_r
    elif hor_r is not None:
        r = hor_r
    elif ver_r is not None:
        r = ver_r

    f.write('\n** final result = ')
    f.write(str(x) + ',' + str(y) + ' | ratio= ' + str(
        np.around(r / iris_r, decimals=2)) + ' iris r = ' + str(iris_r))

    print('final result', x, y, r, np.around(r / iris_r, decimals=2), iris_r)

    return x,y,r

# create mask
# x, y, r = result
# h, w = img.shape
# circle_image = np.full(img.shape[:2], 255, np.uint8)
# circle_image = cv2.circle(circle_image, (int(x), int(y)), int(r), 0, thickness=-1)
# masked_data = cv2.bitwise_not(v_channel, mask=circle_image)
# cv2.imwrite(storing_path + 'ap2_masked_' +file, masked_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('approach', type=int)
    args = parser.parse_args()
    args = vars(args)
    approach = args['approach']
    global storing_path
    storing_path = reading_path + '/' + str(approach) + '/'

    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    RGB = onlyfiles

    f = open(storing_path + "result.txt", "w")

    for file in RGB:
        # get index
        index = re.findall(r"[\d]", file)
        index = ''.join(index)
        f.write('\n\n\nindex ' + index + " --------------------")
        print('\n==================================')
        print('index', index, '---------------')
        print('==================================')

        rgb = cv2.imread(join(input_path, file))
        original_width, original_height = rgb.shape[:2]
        print('original size (w,h): ' + str(original_width) + ' ' + str(original_height))

        f.write('\n finding iris')
        print('[finding iris]')
        crop_area, roi_gray, roi_rgb = ed.extract_eyeball_ROI(rgb)
        h, w = roi_gray.shape[:2]
        iris_r = min(h, w)/2
        if roi_gray is None:
            f.write('\nROI is None')
            print('** ROI is None')
            continue

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        _, _, v_channel = cv2.split(hsv)
        v_channel = tools.image_denoise(v_channel)
        v_channel_inv = cv2.bitwise_not(v_channel)

        v_channel_inv = cv2.equalizeHist(v_channel_inv)
        v_channel_inv = cv2.medianBlur(v_channel_inv, 5)

        result_1 = None
        x, y, r = 0, 0, 0

        if approach == 0:
            result_1 = approach_hough_cnt(v_channel, file, original_width)
            print('result 1', result_1)
            f.write('\nresult 1 ' + str(result_1))

            #select from first two approach first
            if result_1 is not None:
                x, y, r = result_1

            result_2 = appraoch_inv(v_channel_inv, result_1)
            print('\nresult 2', result_2)
            f.write('\nresult 2 ' + str(result_2))

            if result_2 is not None:
                x2, y2, r2 = result_2

                pixel_1, _ = tools.check_roi_pixels(x, y, r, roi_gray)
                pixel_2, _ = tools.check_roi_pixels(x2, y2, r2, roi_gray)
                print('result 1 = ', pixel_1, ' vs result 2 =', pixel_2)

                if pixel_1 > pixel_2:
                    x,y,r = result_2

            #select with the third
            x3, y3, r3 = approach_expand(v_channel_inv, (x, y), 5)

            f.write('\n' + str(x3) + ',' + str(y3) + ',' + str(r3) + ' compare with ' + str(x) + ',' + str(y) + ',' + str(r))
            print(x3, y3, r3, 'compare with', x, y, r)

            pixel_1, _ = tools.check_roi_pixels(x, y, r, roi_gray)
            pixel_3, _ = tools.check_roi_pixels(x3, y3, r3, roi_gray)
            print(pixel_1, pixel_3)
            if pixel_1 > pixel_3 or (x == 0 and y ==0 and r == 0) and r3 is not None:
                x,y,r = x3, y3, r3

        elif approach == 1:
            result_1 = approach_hough_cnt(v_channel, file, original_width)
            print('result 1', result_1)
            f.write('\nresult 1 ' + str(result_1))

            x, y, r = 0, 0, 0
            if result_1 is not None:
                # select from first two approach first
                x, y, r = result_1

        elif approach == 2:
            result_2 = appraoch_inv(v_channel_inv, result_1)
            print('\nresult 2', result_2)
            f.write('\nresult 2 ' + str(result_2))
            if result_2 is not None:
                x, y, r = result_2

        elif approach == 3:
            #select with the third
            result_3 = approach_expand(v_channel_inv, (x, y), 5)
            print('\nresult 3 ', result_3)
            f.write('\nresult 3' + str(result_3))
            if result_3 is not None:
                x, y, r = result_3

        x += crop_area[0] #crop for getting the eyeball

        print('[FINAL RESULT]', x, y, r)
        f.write('\n[FINAL RESULT] ' + str(x) + ',' + str(y) + ',' + str(r)
                + ' ' + str(np.around(r/iris_r, decimals=2)))
        if r == 0: continue

        output = rgb
        output = cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), thickness=2)
        output = cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), thickness=2)
        cv2.imwrite(storing_path + 'final_' + file, output)

    f.close()

# Edge Approach
# edge_inv = cv2.Canny(v_channel_inv, 20, 80)
# edge = cv2.Canny(v_channel, 20, 80)
# edge = cv2.erode(edge, (7, 7), 1)
# edge_inv = cv2.erode(edge_inv, (7, 7), 1)
#
# # edge = cv2.dilate(edge, (3, 3), 5)
# # edge_inv = cv2.dilate(edge_inv, (3, 3), 5)
#
# # edge = cv2.medianBlur(edge, 3)
# # edge_inv = cv2.medianBlur(edge_inv, 3)
#
# combine = cv2.bitwise_or(edge, edge_inv)
#
# cv2.imwrite(storing_path + 'edge_' + file, edge)
# cv2.imwrite(storing_path + 'combine_' + file, combine)
# cv2.imwrite(storing_path + 'edge_inv_' + file, edge_inv)
#
# def check_ROI_first(roi, avg):
#     print('--- check roi', avg)
#     cv2.imwrite(storing_path + 'pupil_roi' + '_' + str(avg) + '_' + file, roi)
#
#     roi = np.array(roi)
#     h, w = roi.shape
#
#     print('roi', w , h)
#     #cols
#     pixels_cols = roi.sum(axis=0)
#     pixels_cols, gradient_2_cols, \
#     maxima_cols, maxima_g2_cols, keeps_cols = tools.get_roi_attributes(pixels_cols, h)
#     print('[COLS] sd', np.std(pixels_cols),'mean', np.nanmean(pixels_cols),
#           'max-min', np.ptp(pixels_cols), maxima_g2_cols)
#
#     #### rows
#     pixels_row = roi.sum(axis=1)
#     pixels_row, gradient_2_row, \
#     maxima_row, maxima_g2_row, keeps_row = tools.get_roi_attributes(pixels_row, w)
#     print('[ROWS] sd', np.std(pixels_row), 'mean', np.nanmean(pixels_row),
#           'max-min', np.ptp(pixels_row), maxima_g2_row)
#
#     # plot
#     fig, (ax1,ax2) = plt.subplots(2)
#     ax1.plot(pixels_cols, label='cols')
#     ax2.plot(gradient_2_cols, label='2nd grad col')
#
#     # ax1.scatter(maxima_cols, pixels_cols[maxima_cols], label='greater col')
#     # ax2.scatter(maxima_g2_cols, gradient_2_cols[maxima_g2_cols], label='g2 greater col')
#     # ax1.axvspan(maxima_g2_cols[0], maxima_g2_cols[1], alpha=0.2, color='yellow')
#
#     ax1.plot(pixels_row, label='rows')
#     ax2.plot(gradient_2_row, label='2nd grad row')
#
#     # ax1.scatter(maxima_row, pixels_row[maxima_row], label='greater row')
#     # ax2.scatter(maxima_g2_row, gradient_2_row[maxima_g2_row], label='g2 less row')
#     # ax1.axvspan(maxima_g2_row[0], maxima_g2_row[1], alpha=0.2, color='pink')
#
#     ax1.legend(loc='best')
#     ax2.legend(loc='best')
#     ax1.grid()
#     ax2.grid()
#     fig.savefig(storing_path + 'roi_plt_' + str(avg) + '_' + file + '.png', dpi=100)
#     # plt.close(fig)
#     print('output fig')
#
#     roi = roi[keeps_row[0]:keeps_row[1], keeps_cols[0]:keeps_cols[1]]
#     cv2.imwrite(storing_path + 'pupil_roi' + '_after_' + str(avg) + '_' + file, roi)
#
#     _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
#     cv2.imwrite(storing_path + 'pupil_roi' + '_after_binary' + str(avg) + '_' + file, binary)
#     print('output binary\n')
#
#     return None
