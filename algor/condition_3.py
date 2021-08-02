import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
from scipy.spatial import distance
import tools
import matplotlib.pyplot as plt
from scipy import signal

condition = 'condition_3' #only works when detected area is smaller than actual, not works on conditon 1
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'
storing_path = reading_path + '/result/'
EDGE = []
CHANNEL = []
RGB = []

avg_iris = 70
PIR_UPPER = 0.65
PIR_LOWER = 0.2

def select_pupil_radius(inputs, iris_x=None, iris_y=None, iris_r=avg_iris):
    if inputs is None or len(inputs) <= 0: return None
    min_d = (float('inf'), None)
    if iris_x is not None and iris_y is not None:
        for cont in inputs:
            x, y, r = cont
            ratio = np.around(r/iris_r, decimals=2)

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
        inputs = [x for x in inputs if np.around(x[2]/iris_r, decimals=2) <= PIR_UPPER
                                                   and np.around(x[2]/iris_r, decimals=2) >= PIR_LOWER and x[0] >= 10 and x[1] >= 10]
        if len(inputs) == 0: return None
        x,y, r = max(inputs, key=lambda x: x[2])

    return (x, y, r)

def select_final(result_1, result_2, iris_result):
    #result 1: result from hough circle, priority
    #result 2: result from contours
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
        #get index
        index = re.findall(r"[\d]", file)
        index = ''.join(index)
        print('\nindex', index)

        #open edge pic
        edge_file = [f for f in EDGE if f.__contains__(index)]
        if len(edge_file) == 0: break
        edge_file = edge_file[0]
        edge = cv2.imread(join(input_path, edge_file), 0)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, (5, 5), 3)

        #open rgb pic
        rgb_file = [f for f in RGB if f.__contains__(index)]
        if len(rgb_file) == 0: break
        rgb_file = rgb_file[0]
        rgb = cv2.imread(join(input_path, rgb_file))

        hsv = cv2.imread(join(input_path, file), 0)

        # ------------------------ find eyeball --------------------------------- #
        #applying hough circles on gray image
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (3, 3))
        # gray = cv2.GaussianBlur(gray, (7, 7), 3)
        gray = cv2.medianBlur(gray, 5)

        _, gray_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(storing_path + 'gray_binary_' + file, gray_binary)

        h, w = gray_binary.shape
        gray_cols = gray_binary.sum(axis=0)
        gray_rows = gray_binary.sum(axis=1)

        gray_pixels = np.add.reduceat(gray_cols, np.arange(0, len(gray_cols), 10))
        minimal = signal.argrelextrema(gray_pixels, np.less)[0]
        iris_x = (minimal[-1] + minimal[0]) / 2
        iris_distance = ((minimal[-1] - minimal[0])/2) * 10

        pre_x = np.where(gray_cols == max(gray_cols))
        pre_y = np.where(gray_rows == max(gray_rows))
        print(max(gray_cols))
        print(minimal, iris_distance, iris_x, (pre_x, pre_y))

        eyeball_circles = cv2.HoughCircles(gray_binary, cv2.HOUGH_GRADIENT, 1, 30,
                                             param1=40, param2=10, minRadius=50, maxRadius=0)
        print(eyeball_circles)

        if eyeball_circles is not None:
            eyeball = [i for i in eyeball_circles[0] if i[2] <= iris_distance and abs(i[1] - (h/2)) <= 15]
            print(eyeball)
            if len(eyeball) > 0:
                eyeball = max(eyeball, key=lambda x: x[2])
                # avg_iris = np.around(np.average((avg_iris, eyeball[2])), decimals=1)

        #draw circle on rgb
        if len(eyeball) > 0:
            iris_x, iris_y, iris_r = eyeball
            print('iris', iris_x, iris_y, iris_r)

        #detect glint, use grayscale
        max_t, min_t = tools.get_threshold(gray)
        print('min t on grayscale:', min_t)
        _, glint = cv2.threshold(gray, min_t, 255, cv2.THRESH_BINARY_INV)
        glint = cv2.morphologyEx(glint, cv2.MORPH_CLOSE, (7, 7), 5)
        cv2.imwrite(storing_path + 'glint_' + file, glint)

        cv2.circle(gray, (iris_x, pre_y[0]), 1, (255, 0, 0), 2)
        cv2.imwrite(storing_path + 'gray_' + file, gray)

        # ------------------------------- find pupil --------------------------------------#
        # hsv = np.where(hsv <= 30, 255, hsv)
        # hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, (7, 7), 7)
        # cv2.imwrite(storing_path + 'hsv_' + file, hsv)

        output = cv2.imread(join(input_path, file))

        combine_glint = cv2.bitwise_and(hsv, hsv, mask=glint)
        combine_glint = np.where(combine_glint == 0, 255, combine_glint) #0 is black, 255 is white
        combine_glint = cv2.erode(combine_glint, (7, 7), 5)
        combine_glint = cv2.erode(combine_glint, (7, 7), 5)

        max_t, min_t = tools.get_threshold(combine_glint)
        # # t = np.amin(combine_glint)
        print('threshold min->', min_t, max_t)
        # _, combine_glint = cv2.threshold(combine_glint,  min_t-20, 255, cv2.THRESH_BINARY)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (7, 7), 7)

        row, col = combine_glint[:2]

        pixels = combine_glint.sum(axis=1)
        peak = np.where(pixels == min(pixels))[0][0]
        pixels = combine_glint[peak]
        gradient = np.gradient(pixels)
        gradient_2 = np.gradient(gradient)

        #plot
        plt.plot(pixels, label='value')
        plt.plot(gradient, label='1st')
        plt.plot(gradient_2, label='2nd')

        plt.legend(loc='best')
        plt.grid()
        plt.savefig(storing_path + 'plt_' + file + '.png', dpi=100)
        plt.close()

        print('peak', peak)
        cv2.line(combine_glint, (0, int(peak)), (int(170), int(peak)), (255, 0, 0), 2)
        cv2.imwrite(storing_path + 'hsv_' + file, combine_glint)

        #detect contour
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = [ cv2.minEnclosingCircle(i) for i in contours]
        # contours = [ [a, b, c] for (a, b), c in contours]

        circle_result, contours_result = None, None

        if contours is not None:
            contours = [ np.around(x) for x in contours]
            contours_result = select_pupil_radius(contours, iris_x, iris_y, iris_r)
            print('contours ' + index + "=", contours_result)

        final_result = select_final(circle_result, contours_result, eyeball)
        if final_result is not None:
            print('\nfinal result => ', final_result, ' |ratio=', final_result[2]/iris_r,'| iris r = ', iris_r)
            x, y, r = final_result
            cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), 2)
            cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), 3)

        if len(eyeball) > 0:
            x, y, r = np.around(eyeball)
            print("eye ball " + index + "=", x, y, r)
            cv2.circle(output, (x, y), int(r), (255, 0, 0), 2)
            cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

        # cv2.imwrite(storing_path + 'thres_' + file, thresh)
        cv2.imwrite(storing_path + file, output)



