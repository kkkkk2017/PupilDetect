import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
from scipy.spatial import distance
import tools
import matplotlib.pyplot as plt
from scipy import signal

condition = 'condition_4'  # only works when detected area is smaller than actual, not works on conditon 1
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'
storing_path = reading_path + '/result/'
EDGE = []
CHANNEL = []
RGB = []

avg_iris = 60
PIR_UPPER = 0.65
PIR_LOWER = 0.2

def get_peak(half_ary, if_back=False):
    if len(half_ary) == 0:
        print('ary is empty, [ERROR]')
        return 0

    peak_value = max(half_ary)
    peak = np.where(half_ary == peak_value)[0]
    if if_back:
        return peak[-1]
    else:
        return peak[0]

def select_pupil_radius(inputs, iris_x=None, iris_y=None, iris_r=avg_iris):
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


if __name__ == '__main__':
    eyeball = []
    iris_x, iris_y = None, None
    iris_r = avg_iris

    for d in ['EDGE', 'CHANNEL', 'RGB']:
        onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))
                     and (f.__contains__(d))]
        exec(d + '=onlyfiles')
        print(d, len(eval(d)))

    for file in CHANNEL:
        # get index
        index = re.findall(r"[\d]", file)
        index = ''.join(index)
        print('\nindex', index)

        # open edge pic
        edge_file = [f for f in EDGE if f.__contains__(index)]
        if len(edge_file) == 0: break
        edge_file = edge_file[0]
        edge = cv2.imread(join(input_path, edge_file), 0)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, (5, 5), 3)

        # open rgb pic
        rgb_file = [f for f in RGB if f.__contains__(index)]
        if len(rgb_file) == 0: break
        rgb_file = rgb_file[0]
        rgb = cv2.imread(join(input_path, rgb_file))

        hsv = cv2.imread(join(input_path, file), 0)

        # ------------------------ find eyeball --------------------------------- #
        # applying hough circles on edge image
        eyeball_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 50,
                                           param1=60, param2=30, minRadius=50, maxRadius=0)
        if eyeball_circles is not None:
            eyeball = [i for i in eyeball_circles[0] if abs(i[2] - avg_iris) <= 3]
            if len(eyeball) > 0:
                eyeball = max(eyeball, key=lambda x: x[2])
                # avg_iris = np.around(np.average((avg_iris, eyeball[2])), decimals=1)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            eyeball_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                               param1=60, param2=30, minRadius=50, maxRadius=0)
            if eyeball_circles is not None:
                eyeball = [i for i in eyeball_circles[0] if abs(i[2] - avg_iris) <= 3]
                if len(eyeball) > 0:
                    eyeball = max(eyeball, key=lambda x: x[2])
                    # avg_iris = np.around(np.average((avg_iris, eyeball[2])), decimals=1)

        # draw circle on rgb
        if len(eyeball) > 0:
            iris_x, iris_y, iris_r = eyeball

        # ------------------------------- find pupil --------------------------------------#
        max_t, min_t = tools.get_threshold(hsv)
        print('threshold =', max_t, "<-max|min->", min_t)

        _, thresh = cv2.threshold(hsv, max_t, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, (11, 11), 5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (7, 7), 7)

        cv2.imwrite(storing_path + 'edge_' + file, edge)
        cv2.imwrite(storing_path + 'thres_' + file, thresh)
        # cv2.imwrite(storing_path + file, output)

        r, c = thresh.shape
        # [0-100]
        cols = thresh.sum(axis=0)
        rows = thresh.sum(axis=1)
        cols = cols / r  # average
        rows = rows / c  # average

        # cols = cols/len(cols)
        cols = signal.savgol_filter(cols, 31, 3)
        rows = signal.savgol_filter(rows, 11, 3)
        cols = cols[15: c-15]
        rows = rows[10: r-10]
        plt.plot(cols, label='col')
        plt.plot(rows, label='row')

        # max_cols, properties = signal.find_peaks(cols, prominence=1, width=10)
        # iris_r = max_cols[0] - max_cols[-1]
        # center_x = (max_cols[0] + max_cols[-1])/2
        # print('center_x', center_x)
        # print(sort)
        # r = abs(sort[0] - sort[1])
        # r = np.around(r)
        # r = int(r)
        # print('new r', r)

        peaks_col, property_col = signal.find_peaks(cols, prominence=50, width=5)
        peaks_row, property_row = signal.find_peaks(rows, prominence=1, width=5)
        print(property_col['widths'], peaks_row)
        widths = [i for i in property_col['widths'] if PIR_LOWER <= i/iris_r <= PIR_UPPER ]
        print(property_row['prominences'])

        print(widths, peaks_row, )
        if len(widths) > 0:
            r = np.around(min(widths))

        x = (peaks_col[-1] + peaks_col[0]) / 2
        y = peaks_row[np.where(property_row['prominences'] == max(property_row['prominences']))[0][0]]
        y += 10
        x += 15

        plt.scatter(peaks_col, cols[peaks_col], label='col peaks')
        plt.scatter(peaks_row, rows[peaks_row], label='row peaks')
        print('final result ', x, y, r, r/iris_r)

        output = cv2.imread(join(input_path, file))
        output = cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), thickness=2)
        cv2.imwrite(storing_path + 'output_' + file, output)

        plt.legend(loc='best')
        plt.grid()
        plt.savefig(storing_path + 'plt_' + file + '.png', dpi=100)
        plt.close()



