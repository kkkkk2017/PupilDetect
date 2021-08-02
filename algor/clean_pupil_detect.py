import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
from scipy.spatial import distance
from scipy import signal
import eyeball_detect as ed
import tools

condition = 'eyeballs' #only works when detected area is smaller than actual, not works on conditon 1
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'
storing_path = reading_path + '/pupil/'

RGB = []
PIR_UPPER = 0.55
PIR_LOWER = 0.25
avg_iris = 61
BLACK_PIXEL_THRESHOLD = 50
WHITE_PIXEL_THRESHOLD = 195
WHITE_PIXEL_SECONDARY_THRESHOLD = 150

#select result by euclidean distance and count pixels of ROI
#roi here is eyeball
def select_by_pd(eyeball, results, pre_x, pre_y, iris_r):
    print('circle result PIR ', [np.around(i[2]/iris_r, decimals=2) for i in results])

    result_set = [check_roi_pixels(int(i[0]), int(i[1]), int(i[2]), eyeball) for i in results]
    result_set = list(sorted(result_set, key=lambda x: (distance.euclidean((x[1][0], x[1][1]), (pre_x, pre_y)), x[0])))

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
    # output = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    print('\n-------[APPROACH HGH + CNT]-----------------------------------')
    height, width = roi.shape
    iris_r = min(height, width)/2

    f.write('\nROI size (w,h) ' + str(width) + ' ' + str(height))
    print('ROI size ' + str(width) + ' ' + str(height))

    pre_x, pre_y = ed.get_preliminary_center(roi)

    dis_to_center = distance.euclidean( (pre_x, pre_y), (width/2,height/2))
    print('dis to centerr =', dis_to_center)

    if np.around(dis_to_center, decimals=0) > 15:
        pre_x = int(width/2)
        pre_y = int(height/2)

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

    _, binary = cv2.threshold(roi, min_t - 20, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (7, 7))

    ## trial 1: hough circle
    # gray_circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=30, minRadius=0, maxRadius=100)

    ##  select the results
    circle_results = tools.hg_n_cnt(binary)
    circle_results = np.around(circle_results)
    f.write('\ncircle reuslt: \n')
    for i in circle_results:
        f.write(str(np.around(i[2] /iris_r, decimals=2)) + '\t')

    #remove the invalid results
    circle_results = [i for i in circle_results
                      if PIR_LOWER <= np.around(i[2] /iris_r, decimals=2) <= PIR_UPPER]

    if len(circle_results) == 0: return None

    final_result = select_by_pd(roi, circle_results, pre_x, pre_x, iris_r)
    print('final result', final_result)
    if final_result is None: return None

    return final_result


def get_breakpoint(all_index, reverse=False):
    if len(all_index) <= 1: return -1

    if reverse:
        all_index = list(reversed(all_index))

    for a, b in zip(all_index[:-1], all_index[1:-2]):
        if abs(b-a) != 1:
            return a

    return all_index[0]


# crop the roi by threshold
def crop_roi_threshold(full_array):
    maxima = np.where(full_array == max(full_array))[0][0]
    # pixels[pixels < WHITE_PIXEL_SECONDARY_THRESHOLD] = 0
    smaller_pixels = np.where(full_array < WHITE_PIXEL_SECONDARY_THRESHOLD)[0]
    break_point_1 = get_breakpoint(smaller_pixels)
    break_point_2 = get_breakpoint(smaller_pixels, reverse=True)

    print('break point', break_point_1, break_point_2, 'maxima index', maxima)
    if break_point_1 == -1 and break_point_2 == -1:
        keeps = (0, len(full_array))
    elif break_point_1 == smaller_pixels[0] and break_point_2 == smaller_pixels[-1]:
        if break_point_1 < maxima:
            keeps = [break_point_2, len(full_array)]
        else:
            keeps = [0, break_point_1]
    else:
        keeps = (break_point_1, break_point_2)
    return keeps


def get_roi_attributes(pixels, avg, crop=True):
    if pixels is None: return  pixels, _, _, _, _

    pixels = pixels / avg
    if crop:
        keeps = crop_roi_threshold(pixels)
        print('keeps', keeps)
        pixels = pixels[keeps[0]: keeps[1]]
    else:
        keeps = [0, len(pixels)]

    if keeps[1] - keeps[0] <= 2:
        return pixels, _, _, _, keeps

    maxima = signal.argrelextrema(pixels, np.greater)[0]

    gradient = np.gradient(pixels)
    gradient_2 = np.gradient(gradient)

    sort = np.argsort(gradient_2)[::-1]
    index_1 = sort.flatten()[0] #get the biggest value

    sub = gradient_2[index_1+1:] #get the partition after global maximum

    if len(sub) == 0: return pixels, gradient_2, _, _, keeps

    sort = np.argsort(sub)[::-1] # get the maximum on the other side
    index_2 = sort.flatten()[0] + index_1 + 1

    return pixels, gradient_2, maxima, list(sorted([index_1, index_2])), keeps


def histogram_on_roi(x, y, r, img):
    roi = img[y - r:y + r, x - r:x + r]
    if roi is None or len(roi) == 0: return 0, 0, 0

    #roi here should be pupil roi
    _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    if binary is None: return 0,0,0

    h, w = binary.shape
    # [0-100]
    cols = binary.sum(axis=0)
    rows = binary.sum(axis=1)

    cols, gradient_col_2, _, _, _= get_roi_attributes(cols, h, False)
    rows, gradient_row_2, _, _, _= get_roi_attributes(rows, w, False)

    middle_row = np.where(rows == min(rows))[0][0]
    middle_col = np.where(cols == min(cols))[0][0]

    peaks_col = [get_peak(gradient_col_2[:middle_col], if_back=True),
                 get_peak(gradient_col_2[middle_col:]) + middle_col]

    peaks_row = [get_peak(gradient_row_2[:middle_row], if_back=True),
                 get_peak(gradient_row_2[middle_row:]) + middle_row]

    assum_x = (peaks_col[1] + peaks_col[0]) / 2
    assum_y = (peaks_row[1] + peaks_row[0]) / 2

    col_r = abs(peaks_col[1] - peaks_col[0]) / 2
    row_r = abs(peaks_row[1] - peaks_row[0]) / 2

    assum_x += (x-r)
    assum_y += (y-r)

    print('assume (x,y)', assum_x, assum_y, col_r, row_r)
    f.write('\nassume (x,y, col r, row r) ' + str([assum_x, assum_y, col_r, row_r]))

    return assum_x, assum_y, max(col_r, row_r)

#check if the roi still have some areas to crop
def crop_small_roi(roi, avg):

    roi = np.array(roi)
    h, w = roi.shape

    # cols
    pixels_cols = roi.sum(axis=0)
    pixels_cols, _, _, _, keeps_cols = get_roi_attributes(pixels_cols, h)
    #### rows
    pixels_row = roi.sum(axis=1)
    pixels_row, _, _, _, keeps_row = get_roi_attributes(pixels_row, w)

    #crop by pixel limit
    roi = roi[keeps_row[0]:keeps_row[1], keeps_cols[0]:keeps_cols[1]]
    h, w = roi.shape

    return (keeps_cols[0], keeps_row[0], max(h, w))


#count avgerage pixel values in roi and crop the roi if needed.
def check_roi_pixels(x, y, r, eyeball, mode=0):
    #mode 0 : black
    #mode 1 : white
    x = int(x)
    y = int(y)
    r = int(r)
    roi = eyeball[y-r:y+r, x-r:x+r]

    if roi is None or len(roi) == 0:
        if mode == 0:
            return 255, (x, y, r)
        else:
            return 0, (x, y, r)

    avg = np.average(roi)
    avg = np.around(avg, decimals=0)

    if mode == 0:
        if avg > BLACK_PIXEL_THRESHOLD: return 255, (x, y, r)
    else:
        if avg < WHITE_PIXEL_SECONDARY_THRESHOLD: return 0, (x, y, r)
        if WHITE_PIXEL_THRESHOLD > avg >= WHITE_PIXEL_SECONDARY_THRESHOLD:
            (crop_x, crop_y, crop_r) = crop_small_roi(roi, avg) #crop roi
            return check_roi_pixels(x+crop_x, y+crop_y, crop_r, eyeball, mode)

    return avg, (x, y, r)


def approach_inv(eyeball, result):
    #roi here is eyeball

    # output = cv2.cvtColor(eyeball, cv2.COLOR_GRAY2RGB)
    h, w = eyeball.shape[:2]
    iris_r = min(h, w)

    print('\n-------[APPROACH INV]-----------------------------------')
    max_t, min_t = tools.get_threshold(eyeball)
    f.write('\nthreshold = ' + str(max_t) + "<-max|min->" + str(min_t))

    _, binary = cv2.threshold(eyeball, max_t + 20, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, (7, 7))

    circle_results = tools.hg_n_cnt(binary)
    ##  select the results
    circle_results = np.append(circle_results, [result], axis=0)
    circle_results = np.around(circle_results)

    f.write('\ncircle reuslt: \n')
    for i in circle_results:
        f.write(str(np.around(i[2] /iris_r, decimals=2)) + '\t')

    circle_results = [i for i in circle_results
                      if PIR_LOWER <= np.around(i[2] /iris_r, decimals=2) <= PIR_UPPER]

    if len(circle_results) == 0: return None

    distance_set = [(check_roi_pixels(int(i[0]), int(i[1]), int(i[2]), eyeball, mode=1))
                    for i in circle_results]
    distance_set = [ (value[0], histogram_on_roi(value[1][0], value[1][1], value[1][2], eyeball))
                     for value in distance_set]

    distance_set = list(sorted(distance_set, key=lambda x: x[0], reverse=True))

    f.write('\ndistance set, avg pixel = ' + str(distance_set))
    final_result = distance_set[0]

    if final_result[0] == 0: return None

    final_result = final_result[1]

    f.write('\n** final result = ')
    f.write(str(final_result[0]) + ',' + str(final_result[1]) + ' | ratio= ' + str(np.around(final_result[2] / iris_r, decimals=2)))
    print('final result', final_result[0], final_result[1], '| ratio=', np.around(final_result[2] / iris_r, decimals=2))

    return final_result


#create mask
#x, y, r = result
# h, w = img.shape
# circle_image = np.full(img.shape[:2], 255, np.uint8)
# circle_image = cv2.circle(circle_image, (int(x), int(y)), int(r), 0, thickness=-1)
# masked_data = cv2.bitwise_not(v_channel, mask=circle_image)
# cv2.imwrite(storing_path + 'ap2_masked_' +file, masked_data)
def get_peak(half_ary, if_back=False):
    if len(half_ary) == 0:
        print('ary is empty, [ERROR]')
        return 0

    # peaks = signal.find_peaks(half_ary)[0]
    peak_value = max(half_ary)
    peak = np.where(half_ary == peak_value)[0]
    if if_back:
        return peak[-1]
    else:
        return peak[0]

# check if the radius is valid
def valid_radius(pupil_r): #pir
    pupil_r = np.around(pupil_r, decimals=2)
    if pupil_r <= PIR_LOWER or pupil_r >= PIR_UPPER: return False
    return True


if __name__ == '__main__':
    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    RGB = onlyfiles

    f = open( storing_path + "result.txt", "w")

    for file in RGB:
        #get index
        index = re.findall(r"[\d]", file)
        index = ''.join(index)
        f.write('\n\n\nindex ' + index + " --------------------")
        print('\n==================================')
        print('index', index, '---------------')
        print('==================================')

        rgb = cv2.imread(join(input_path, file))
        original_height, original_width = rgb.shape[:2]
        print('original size (w,h): ' + str(original_width) + ' ' + str(original_height))

        #find iris
        f.write('\n finding iris')
        print('[finding iris]')
        roi_gray, roi_rgb = ed.extract_eyeball_ROI(rgb)

        if roi_gray is None:
            f.write('\nROI is None')
            print('** ROI is None')
            continue

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        _, _, v_channel = cv2.split(hsv)
        v_channel = ed.image_denoise(v_channel)
        v_channel_inv = cv2.bitwise_not(v_channel)

        v_channel_inv = cv2.equalizeHist(v_channel_inv)
        v_channel_inv = cv2.medianBlur(v_channel_inv, 5)

        result_1 = approach_hough_cnt(v_channel, file, original_width)
        print('result 1', result_1)
        if result_1 is None: continue

        result_2 = approach_inv(v_channel_inv, result_1)
        print('\nresult 2', result_2)
        if result_2 is not None:
            x1, y1, r1 = result_1
            x2, y2, r2 = result_2

            output = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            output = cv2.circle(output, (int(x1), int(y1)), int(r1), (0, 0, 255), thickness=2)
            output = cv2.circle(output, (int(x1), int(y1)), 2, (0, 0, 255), thickness=2)
            output = cv2.circle(output, (int(x2), int(y2)), int(r2), (0, 255, 0), thickness=2)
            output = cv2.circle(output, (int(x2), int(y2)), 2, (0, 255, 0), thickness=2)
            cv2.imwrite(storing_path + 'compare_' + file, output)

            print('result 1 = ', check_roi_pixels(x1, y1, r1, roi_gray), ' vs result 2 =', check_roi_pixels(x2, y2, r2, roi_gray))
            if check_roi_pixels(x1, y1, r1, roi_gray)[0] <= check_roi_pixels(x2, y2, r2, roi_gray)[0]:
                output = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                output = cv2.circle(output, (int(x1), int(y1)), int(r1), (255, 0, 0), thickness=2)
                output = cv2.circle(output, (int(x1), int(y1)), 2, (255, 0, 0), thickness=2)
                cv2.imwrite(storing_path + 'final_' + file, output)
            else:
                output = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                output = cv2.circle(output, (int(x2), int(y2)), int(r2), (255, 0, 0), thickness=2)
                output = cv2.circle(output, (int(x2), int(y2)), 2, (255, 0, 0), thickness=2)
                cv2.imwrite(storing_path + 'final_' + file, output)
        else:
            x1, y1, r1 = result_1
            output = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            output = cv2.circle(output, (int(x1), int(y1)), int(r1), (255, 0, 0), thickness=2)
            output = cv2.circle(output, (int(x1), int(y1)), 2, (255, 0, 0), thickness=2)
            cv2.imwrite(storing_path + 'final_' + file, output)

    f.close()
