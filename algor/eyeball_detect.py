import cv2
from os import listdir, path
from os.path import isfile, join
import numpy as np
import re
import tools
import matplotlib.pyplot as plt
from scipy import signal

condition = 'eyeballs' #only works when detected area is smaller than actual, not works on conditon 1
reading_path = path.expanduser('~/Desktop/test_eye_images/' + condition)
input_path = reading_path + '/input/'
storing_path = reading_path + '/result/'

RGB = []

def select_remove_range(gradient_2, gradient, values, pre_x):

    #get the low peaks of second gradient
    candidates = signal.argrelextrema(gradient_2, np.less)[0]

    #get the global minima of the values
    global_min = np.where(values == min(values))[0][0]
    middle = max(global_min, pre_x)
    if middle not in candidates:
        candidates = np.append(candidates, middle)

    grad_min = np.where(gradient == min(gradient))[0][0]
    grad_max = np.where(gradient == max(gradient))[0][0]
    if grad_min not in candidates:
        candidates = np.append(candidates, grad_min)
    if grad_max not in candidates:
        candidates = np.append(candidates, grad_max)

    #sort the array
    #array with low peaks of 2nd gradient and (global min of original values or pre x)
    candidates = np.sort(candidates)
    print(candidates)
    index = np.where(candidates == middle)[0][0]
    minima_values = gradient_2[candidates]

    global_min_g2_index = np.where(minima_values == min(minima_values))[0][0]
    global_min_g2 = candidates[global_min_g2_index]

    if len(candidates) == 3:
        return [candidates[0], candidates[2]]

    first = global_min_g2
    second = middle

    if first < middle:
        crop_values = minima_values[index:]
        second_index = np.where(crop_values == min(crop_values))[0][0]
        second = candidates[second_index+index]

    else:
        crop_values = minima_values[:index]
        if len(crop_values) > 0:
            second_index = np.where(crop_values == min(crop_values))[0][0]
            second = candidates[second_index]

    return list(sorted([first, second])), candidates


def histogram_on_binary(gray_binary, pre_x, if_output=False, file=None):
    h, w = gray_binary.shape
    pre_x = int(np.around(pre_x/10, decimals=0))

    gray_cols = gray_binary.sum(axis=0)
    gray_cols = gray_cols/h
    gray_pixels = np.add.reduceat(gray_cols, np.arange(0, len(gray_cols), 10)) # combine each 10 columns
    gray_pixels = gray_pixels/10

    gradient = np.gradient(gray_pixels)
    gradient_2 = np.gradient(gradient)

    gray_pixels = signal.savgol_filter(gray_pixels, 11, 3)
    gradient = signal.savgol_filter(gradient, 7, 3)
    gradient_2 = signal.savgol_filter(gradient_2, 7, 3)

    maxima = signal.argrelextrema(gray_pixels, np.greater)[0]
    minimal = signal.argrelextrema(gradient, np.less)[0]
    minimal_g2 = signal.argrelextrema(gradient_2, np.less)[0]
    # print(maxima, gray_pixels[maxima])

    g_peaks = signal.find_peaks(gradient)[0]

    #ignore the area out of ROI (potential large white area)
    ignore_area, candidates = select_remove_range(gradient_2, gradient, gray_pixels, pre_x)
    print('ignore area', ignore_area)

    iris_x = (ignore_area[-1] + ignore_area[0]) / 2
    iris_distance = ((ignore_area[-1] - ignore_area[0]) / 2)
    print('iris x|distance', iris_x, iris_distance)

    if if_output and file is not None:
        # plot
        plt.plot(gray_pixels, label='columns')
        plt.plot(gradient, label='1st grad')
        plt.plot(gradient_2, label='2nd grad')

        plt.scatter(pre_x, gray_pixels[pre_x])
        plt.scatter(maxima, gray_pixels[maxima], label='greater')
        plt.scatter(minimal, gradient[minimal], label='g less')
        plt.scatter(minimal_g2, gradient_2[minimal_g2], label='g2 less')
        plt.scatter(candidates, gray_pixels[candidates], label='candidates')

        # x = int(np.around(pre_x, decimals=0))?
        plt.scatter(pre_x, gradient[pre_x], label='pre x')

        plt.scatter(g_peaks, gradient[g_peaks], label='grad peaks')
        # print('min gradient | max gradient', g_peaks)
        # plt.axvspan(selected_peaks[0], selected_peaks[-1], alpha=0.1, color='red')
        # plt.axvspan(selected_max[0], selected_max[-1], alpha=0.1, color='yellow')
        plt.axvline(iris_x, -50, 255)
        plt.axvspan(ignore_area[0], ignore_area[1], alpha=0.1, color='yellow')

        plt.legend(loc='best')
        plt.grid()
        plt.savefig(storing_path + 'plt_' + file + '.png', dpi=100)
        plt.close()

    return iris_x*10, iris_distance*10

def extract_eyeball_ROI(rgb, if_output=False, file=None):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = tools.image_denoise(gray)

    pre_x, pre_y = tools.get_preliminary_center(gray)
    print('pre center', pre_x, pre_y)

    if if_output and file is not None:
        cv2.circle(rgb, (pre_x, pre_y), 2, (0, 0, 255), 3)  # the red dot indicate the preliminary center
        cv2.imwrite(storing_path + 'pre_' + file, rgb)

    _, gray_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_CLOSE, (7, 7))

    if if_output and file is not None:
        cv2.imwrite(storing_path + 'gray_binary_' + file, gray_binary)

    iris_x, iris_distance = histogram_on_binary(gray_binary, pre_x, if_output, file)
    print('on binary image (x|radius):', iris_x, iris_distance)

    try:
        ROI_gray = gray[:, int(iris_x - iris_distance):int(iris_x + iris_distance)]
        ROI_rgb = rgb[:, int(iris_x - iris_distance):int(iris_x + iris_distance)]
        if if_output and file is not None:
            cv2.imwrite(storing_path + 'extract_' + file, ROI_gray)
        return [iris_x - iris_distance, iris_x + iris_distance], ROI_gray, ROI_rgb
    except:
        pass

    return None, None, None


if __name__ == '__main__':
    eyeball = []
    iris_x, iris_y = None, None
    iris_r = 65

    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    RGB = onlyfiles

    for file in RGB:
        #get index
        index = re.findall(r"[\d]", file)
        index = ''.join(index)
        print('\n------- index', index, ' ------------- ')

        rgb = cv2.imread(join(input_path, file))
        extract_eyeball_ROI(rgb, True, file)



