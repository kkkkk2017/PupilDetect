import numpy as np
from scipy.spatial import distance
import cv2
from scipy import signal

PIR_UPPER = 0.55
PIR_LOWER = 0.25
AVG_IRIS = 61
BLACK_PIXEL_THRESHOLD = 50
WHITE_PIXEL_THRESHOLD = 195
WHITE_PIXEL_SECONDARY_THRESHOLD = 150

def if_radius_valid(pupil_r, iris_r):
    if pupil_r is None: return False
    return PIR_LOWER <= np.around(pupil_r / iris_r, decimals=2) <= PIR_UPPER


def image_denoise(gray):
    gray = cv2.equalizeHist(gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (3, 3))
    return gray


def get_preliminary_center(gray_img):
    # applying historygram technique to find eyeball
    h, w = gray_img.shape
    gray_cols = gray_img.sum(axis=0) #horizontal
    gray_rows = gray_img.sum(axis=1) #vertical
    gray_cols = gray_cols/h #find x, height
    gray_rows = gray_rows/w #find y, width

    # preliminary localization of center, C1
    pre_x = np.where(gray_cols == min(gray_cols))[0][-1]
    pre_y = np.where(gray_rows == min(gray_rows))[0][0]

    return pre_x, pre_y


def get_threshold(gray, range_x=15, range_r=0):
    if gray is None: return 0, 0

    start = np.around(range_x-range_r, decimals=0)
    end = np.around(range_x+range_r, decimals=0)
    cols = []
    row, col = gray.shape

    if range_x == 15:
        end = col-15
    # print(start, ' -> ', end)

    for a in np.arange(start, end):
        a = int(a)
        sum = []
        for b in range(row):
            if gray[b][a] != 0 or gray[b][a] != 255:
                sum.append(gray[b][a])

        cols.append(np.mean(sum))

    if len(cols) == 0: return 0, 0

    return np.around(max(cols), decimals=0), np.around(min(cols), decimals=0)


# hough circle and contour detect
def hg_n_cnt(binary_img, param1=20, param2=10, minRadius=0, maxRadius=100):
    binary_circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    contour_circles, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = [cv2.minEnclosingCircle(i) for i in contour_circles]
    contour_circles = [ (a, b, c) for (a, b), c in contour_circles]

    if contour_circles is not None and binary_circles is not None:
        circle_results = np.append(contour_circles, binary_circles[0], axis=0)
    else:
        circle_results = contour_circles if contour_circles is not None else binary_circles[0]
        # circle_results = binary_circles

    return circle_results


#count avgerage pixel values in roi and crop the roi if needed.
def check_roi_pixels(x, y, r, eyeball, mode=0, recur=False, times=3):
    #mode 0 : black
    #mode 1 : white
    if times == 0:
        if mode == 0:
            return 255, (x, y, r)
        else:
            return 0, (x, y, r)

    if r is None:
        if mode == 0:
            return 255, (x, y, r)
        else:
            return 0, (x, y, r)

    x = int(x)
    y = int(y)
    r = int(r)

    north = y-r
    east = x-r
    if north < 0: north = 0
    if east < 0: east = 0

    roi = eyeball[north:y+r, east:x+r]

    if roi is None or len(roi) == 0:
        if mode == 0:
            return 255, (x, y, r)
        else:
            return 0, (x, y, r)

    avg = np.average(roi)
    avg = np.around(avg, decimals=0)

    if mode == 0:
        if avg > BLACK_PIXEL_THRESHOLD:
            return 255, (x, y, r)
    else:
        if avg < WHITE_PIXEL_SECONDARY_THRESHOLD: return 0, (x, y, r)
        if recur:
            if WHITE_PIXEL_THRESHOLD > avg >= WHITE_PIXEL_SECONDARY_THRESHOLD:
                (crop_x, crop_y, crop_r) = crop_small_roi(roi, avg) #crop roi
                return check_roi_pixels(x+crop_x, y+crop_y, crop_r, eyeball, mode, recur, times-1)

    return avg, (x, y, r)

#check if the roi still have some areas to crop
def crop_small_roi(roi, avg, storing_path=None):
    if storing_path is not None:
        cv2.imwrite(storing_path + 'pupil_roi' + '_' + str(avg) + '.jpg', roi)

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
    if storing_path is not None:
        cv2.imwrite(storing_path + 'pupil_roi' + '_after_' + str(avg) + '.jpg', roi)

    return (keeps_cols[0], keeps_row[0], max(h, w))


def get_roi_attributes(pixels, avg, crop=True):
    if pixels is None: return  pixels, [0], None, None, None
    pixels = pixels / avg
    if crop:
        keeps = crop_roi_threshold(pixels)
        # print('keeps', keeps)
        pixels = pixels[keeps[0]: keeps[1]]
    else:
        keeps = [0, len(pixels)]

    if keeps[1] - keeps[0] <= 2:
        return  pixels, [0], None, None, keeps

    maxima = signal.argrelextrema(pixels, np.greater)[0]

    gradient = np.gradient(pixels)
    gradient_2 = np.gradient(gradient)

    sort = np.argsort(gradient_2)[::-1]
    index_1 = sort.flatten()[0] #get the biggest value

    sub = gradient_2[index_1+1:] #get the partition after global maximum

    if len(sub) == 0: return pixels, gradient_2, None, None, keeps

    sort = np.argsort(sub)[::-1] # get the maximum on the other side
    index_2 = sort.flatten()[0] + index_1 + 1

    return pixels, gradient_2, maxima, list(sorted([index_1, index_2])), keeps


# crop the roi by threshold
def crop_roi_threshold(full_array):
    maxima = np.where(full_array == max(full_array))[0][0]
    # pixels[pixels < WHITE_PIXEL_SECONDARY_THRESHOLD] = 0
    smaller_pixels = np.where(full_array < WHITE_PIXEL_SECONDARY_THRESHOLD)[0]
    break_point_1 = get_breakpoint(smaller_pixels)
    break_point_2 = get_breakpoint(smaller_pixels, reverse=True)

    # print('break point', break_point_1, break_point_2, 'maxima index', maxima)
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


def get_breakpoint(all_index, reverse=False):
    if len(all_index) <= 1: return -1

    if reverse:
        all_index = list(reversed(all_index))

    for a, b in zip(all_index[:-1], all_index[1:-2]):
        if abs(b-a) != 1:
            return a

    return all_index[0]


def get_peak(half_ary, if_back=False):
    if len(half_ary) == 0:
        return 0

    # peaks = signal.find_peaks(half_ary)[0]
    peak_value = max(half_ary)
    peak = np.where(half_ary == peak_value)[0]
    if if_back:
        return peak[-1]
    else:
        return peak[0]


# =====================================
def select_pupil_radius(inputs, iris_x=None, iris_y=None, iris_r=AVG_IRIS,
                        PIR_UPPER=PIR_UPPER, PIR_LOWER=PIR_LOWER):

    if inputs is None or len(inputs) <= 0: return None
    min_d = (float('inf'), None)
    if iris_x is not None and iris_y is not None:
        for cont in inputs:
            x, y, r = cont
            ratio = np.around(r / iris_r, decimals=2)

            if ratio <= PIR_LOWER or ratio >= PIR_UPPER: continue
            if x <= 10 or y <= 10: continue
            d = distance.euclidean((x, y), (iris_x, iris_y))
            d = np.around(d)

            if (d >= iris_r): continue
            if (d < min_d[0]):
                min_d = (d, (x, y, r))

        if min_d[1] is None: return None
        x, y, r = min_d[1]
    else:
        inputs = [x for x in inputs if np.around(x[2] / iris_r, decimals=2) <= PIR_UPPER
                  and np.around(x[2] / iris_r, decimals=2) >= PIR_LOWER and x[0] >= 10 and x[1] >= 10]
        if len(inputs) == 0: return None
        x, y, r = max(inputs, key=lambda x: x[2])

    return (x, y, r)


def select_final(result_1, result_2, iris_result):
    # result 1: result from hough circle, priority
    # result 2: result from contours
    if result_1 is None and result_2 is None: return None
    print(result_1, result_2, iris_result)

    if result_1 is not None and result_2 is not None and len(iris_result) > 0:
        x1, y1, r1 = result_1
        x2, y2, r2 = result_2
        xi, yi, ri = iris_result
        d1 = distance.euclidean((x1, y1), (xi, yi))
        d2 = distance.euclidean((x2, y2), (xi, yi))
        if d1 == d2:
            return max((result_1, result_2), key=lambda x: x[2])
        elif d1 < d2:
            return result_1
        else:
            return result_2

    elif result_1 is not None:
        return result_1
    elif result_2 is not None:
        return result_2
    else:
        return max((result_1, result_2), key=lambda x: x[2])

