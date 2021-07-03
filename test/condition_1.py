import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
from scipy.spatial import distance
import tools

condition = 'condition_1'
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'
storing_path = reading_path + '/result/'
EDGE = []
CHANNEL = []
RGB = []

avg_iris = 60
PIR_UPPER = 0.5
PIR_LOWER = 0.15

def select_pupil_radius(inputs, iris_x=None, iris_y=None, iris_r=avg_iris):
    if inputs is None or len(inputs) <= 0: return None
    min_d = (float('inf'), None)

    if iris_x is not None and iris_y is not None:
        for cont in inputs:
            x, y, r = cont
            ratio = np.around(r/iris_r, decimals=2)
            if ratio <= PIR_LOWER or ratio >= PIR_UPPER: continue
            d = distance.euclidean((x, y), (iris_x, iris_y))
            d = np.around(d)
            if (d >= iris_r): continue
            if (d < min_d[0]): min_d = (d, (x, y, r))

        if min_d[1] is None: return (0, 0, 0)
        x, y, r = min_d[1]
    else:
        x,y, r = max(inputs, key=lambda x: x[2] if np.around(x[2]/iris_r, decimals=2) <= PIR_UPPER
                                                   and np.around(x[2]/iris_r, decimals=2) >= PIR_LOWER else 0)

    return (x, y, r)

def select_final(result_1, result_2, iris_result):
    #result 1: result from hough circle, priority
    #result 2: result from contours

    if iris_result is not None:
        return max((result_1, result_2), key=lambda x: x[2])
    else:
        if result_1 is not None and result_2 is not None:
            x1, y1, r1 = result_1
            x2, y2, r2 = result_2
            xi, yi, ri = iris_result
            d1 = distance.euclidean((x1, y1), (xi, yi))
            d2 = distance.euclidean((x2, y2), (xi, yi))
            if d1 == d2: return max((result_1, result_2), key=lambda x: x[2])
            elif d1 < d2: return result_1
            else: return result_2

        elif result_1 is not None:
            return result_1
        elif result_2 is not None:
            return result_2
        else:
            return None


if __name__ == '__main__':

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
        #applying hough circles on edge image
        eyeball_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=60, param2=30, minRadius=50, maxRadius=0)

        if eyeball_circles is not None:
            eyeball = [i for i in eyeball_circles[0] if abs(i[2] - avg_iris) <= 3 ]
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

        #draw circle on rgb
        if len(eyeball) > 0:
            iris_x, iris_y, iris_r = eyeball

        # ------------------------------- find pupil --------------------------------------#
        t = tools.get_threshold(hsv)
        print('threshold =', t)

        # global average pixel value of rgb pictures
        avg = np.average(rgb)
        avg = np.around(avg, decimals=0)
        print('global avg =', avg)

        if avg > 99:
            _, thresh = cv2.threshold(hsv, 10, 255, cv2.THRESH_BINARY_INV)
            # detect pupil
            circles = cv2.HoughCircles(hsv, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=25, minRadius=0, maxRadius=0)
        else:
            _, thresh = cv2.threshold(hsv, t, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (7, 7), 7)
            # detect pupil
            circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=30, minRadius=0, maxRadius=100)

        thresh = cv2.dilate(thresh, (5, 5), 2)
        output = cv2.imread(join(input_path, file))

        #detect contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [ cv2.minEnclosingCircle(i) for i in contours]
        contours = [ [a, b, c] for (a, b), c in contours]

        if circles is not None:
            circles = [ np.around(np.array(x).flatten()) for x in circles[0]]
            circle_result = select_pupil_radius(circles, iris_x, iris_y, iris_r)
            print("circle " + index + "=", circle_result)

        if contours is not None:
            contours = [ np.around(x) for x in contours]
            contours_result = select_pupil_radius(contours, iris_x, iris_y, iris_r)
            print('contours ' + index + "=", contours_result)

        final_result = select_final(circle_result, contours_result, eyeball)
        print('\nfinal result => ', final_result, ' |ratio=', final_result[2]/iris_r,'| final average iris r = ', iris_r)
        x, y, r = final_result
        cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), 2)
        cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), 3)

        if len(eyeball) > 0:
            x, y, r = np.around(eyeball)
            print("eye ball " + index + "=", x, y, r)
            cv2.circle(output, (x, y), int(r), (255, 0, 0), 2)
            cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

        cv2.imwrite(storing_path + 'edge_' + file, edge)
        cv2.imwrite(storing_path + 'thres_' + file, thresh)
        cv2.imwrite(storing_path + file, output)



