import cv2
import numpy as np
import os.path as path
import dlib
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from scipy.signal import find_peaks, savgol_filter

face_file = path.join(path.dirname(__file__), 'face_landmarks.dat')
predictor = dlib.shape_predictor(face_file)
face_detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

UPPER_LIMIT = 50
LOWER_LIMIT = 10 #pupil radius limit

def get_shape(gray, rect):
    shape = predictor(gray, rect)
    shape = imutils.face_utils.shape_to_np(shape)
    return shape

def resize(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=100*3, height=75*3, inter=cv2.INTER_CUBIC)
    return image

def show_image(imgs):
    for i, v in enumerate(imgs):
        cv2.imshow(str(i), v)


def convert_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1]
    img_hsv[:, :, 2] = img_hsv[:, :, 2]

    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr, img_hsv

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # vertical
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

def handle_blink(shape):
    # detecting blink
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return np.around(ear, decimals=2)

def get_radius(depth, center):
    # print('center', center, depth[center])
    center = int(np.around(center))
    # print('depth', depth)
    pnt1 = None
    pnt2 = None

    half_1 = list(reversed(depth[:center]))
    half_2 = depth[center:]

    for i, v in enumerate(half_1):
        if (v == 0 or half_1[i+1] > v) and v != depth[center]:
            pnt1 = center - i
            if pnt1 >= 3:
                break

    for i, v in enumerate(half_2):
        if (v == 0 or half_2[i+1] > v) and v != depth[center]:
            pnt2 = i
            if pnt2 >= 3:
                break

    if pnt1 is None and pnt2 is None: return None
    # print(pnt1, pnt2)
    return min(pnt1, pnt2)


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

# check if the radius is valid
def valid_radius(pupil_r, iris_r):
    if pupil_r <= LOWER_LIMIT or pupil_r >= UPPER_LIMIT: return False

    ratio = pupil_r/iris_r
    # print('ratio', ratio, pupil_r, iris_r)
    if ratio >= 0.2 and ratio <= 0.6: return True
    return False

def smoothing(gray_ary, pixel_ary):
    threshold = max(pixel_ary)/2
    for i in range(len(pixel_ary)):
        if pixel_ary[i] <= threshold and pixel_ary[i] != 0:
            pixel_ary[i] = 0
            gray_ary[i] = 0
    return gray_ary, pixel_ary

# step 0: detect the approximate x, y and threshold
def preprocess(gray, side=None):
    # print('side', side)
    # print('\n[STEP]', step)
    rows = []
    cols = []
    row_pixels = []
    col_pixels = []  # white pixels

    r, c = gray.shape
    # [0-100]
    for a in range(c):
        sum = []
        for b in range(r):
            sum.append(gray[b][a])

        cols.append(np.mean(sum))
        col_pixels.append(len([i for i in sum if i != 0]))

    for v in gray:
        rows.append(np.mean(v))
        row_pixels.append(len([i for i in v if i != 0]))

    x = np.where(cols == min(cols))
    x = int(x[0].mean())
    y = np.where(rows == max(rows))
    y = int(y[0].mean())

    peaks_col = [get_peak(cols[:x]), get_peak(cols[x:])]
    r = peaks_col[-1] - peaks_col[0]
    return (x, y, abs(r)), min(cols)


def get_middle(pixel_ary):
    nzeros = np.where(pixel_ary != 0)
    nzeros = sorted(nzeros)

    # for i in range(len(pixel_ary) - 1):
    #     if pixel_ary[i] == 0 and pixel_ary[i + 1] != 0:
    #         length.append(i + 1)
    #     elif pixel_ary[i] != 0 and pixel_ary[i + 1] == 0:
    #         length.append(i)
    #         break
    # if len(length) == 0:
    #     return int( len(pixel_ary) / 2)
    return int((nzeros[-1] + nzeros[0]) / 2)

def remove_edge(pix_ary, ary):
    peak = pix_ary.index(max(pix_ary))
    ary[peak] = 0
    pix_ary[peak] = 0

    peak = pix_ary.index(max(pix_ary))
    print(peak)
    ary[peak] = 0
    pix_ary[peak] = 0

    return ary

def get_points(ary):
    peaks, properties = find_peaks(ary, width=LOWER_LIMIT, prominence=1)
    prom = properties['prominences']
    if len(prom) == 0: return 0, 0
    # print('peaks', peaks)
    # print('prominence', prom, max(prom))
    p = np.where(prom == max(prom))[0][0]
    # print(properties['widths'], p, properties['widths'][p]/2)
    r = properties['widths'][p]/2
    # print(peaks[p], r)
    return peaks[p], r

# step 1: detect the quality of the blur image, adjust the most likely radius
def find_iris(gray, time=None, side=None, output_pic=False, storing_path=None):

    r, c = gray.shape
    # [0-100]
    cols = gray.sum(axis=0)
    rows = gray.sum(axis=1)
    cols = cols/r #average
    rows = rows/c #average

    try:
        gradient_row = np.gradient(rows)
        gradient_col = np.gradient(cols)
        gradient_col_2 = np.gradient(gradient_col)
        gradient_row_2 = np.gradient(gradient_row)

    except:
        return None

    col_p, col_r = get_points(cols)
    row_p, row_r = get_points(rows)

    peaks_col = (get_peak(gradient_col_2[:col_p], if_back=True),
                 get_peak(gradient_col_2[col_p:])+col_p )
    peaks_row = (get_peak(gradient_row_2[:row_p], if_back=True),
                 get_peak(gradient_row_2[row_p:])+row_p )

    cr = abs(peaks_col[1] - peaks_col[0])/2
    rr = abs(peaks_row[1] - peaks_row[0])/2
    x = (peaks_col[1] + peaks_col[0])/2
    y = (peaks_row[1] + peaks_row[0])/2

    print('col', col_p, x, col_r, cr)
    print('row', row_p, y, row_r, rr)
    return (x-max(cr, rr), x+max(cr, rr))


# step 1: detect the quality of the blur image, adjust the most likely radius
def process(gray, compare_pupil=None, approx_r=24, iris_r=60,
            time=None, side=None, output_pic=False, storing_path=None):

    r, c = gray.shape
    # [0-100]
    cols = gray.sum(axis=0)
    rows = gray.sum(axis=1)
    cols = cols/r #average
    rows = rows/c #average

    try:
        gradient_row = np.gradient(rows)
        gradient_col = np.gradient(cols)
        gradient_col_2 = np.gradient(gradient_col)
        gradient_row_2 = np.gradient(gradient_row)

    except:
        return None, None, (None, None)

    # if output_pic:
        # plot
        # fig, axes = plt.subplots(3, 2)
        # axes[0][0].plot([i for i in cols])
        # axes[0][1].plot([i for i in rows])
        # # axes[1][0].bar([i for i in range(len(col_pixels))], col_pixels)
        # # axes[1][1].bar([i for i in range(len(row_pixels))], row_pixels)
        #
        # axes[2][0].plot(gradient_col_2)
        # axes[2][1].plot(gradient_row_2)
        #
        # axes[0][0].set_xticks([i for i in range(0, len(cols), 30)])
        # axes[0][1].set_xticks([i for i in range(0, len(rows), 10)])
        # axes[2][0].set_xticks([i for i in range(0, len(cols), 30)])
        # axes[2][1].set_xticks([i for i in range(0, len(rows), 10)])
        #
        # axes[0][0].set_ylabel('avg gray value')
        # axes[1][0].set_ylabel('num. of non-0 pixels')
        # axes[2][0].set_ylabel('2nd derivative')
        # axes[2][0].set_xlabel('horizontal (columns) ')
        # axes[2][1].set_xlabel('vertical (rows)')
        #
        # for a in range(3):
        #     for b in range(2):
        #         axes[a][b].grid()

    # middle_col = get_middle(col_pixels)
    col_p, col_r = get_points(cols)
    row_p, row_r = get_points(rows)

    # middle_row = np.where(rows == max(rows))[0][0]
    # middle_col = np.where(cols == max(cols))[0][0]

    peaks_col = [get_peak(gradient_col_2[:col_p], if_back=True),
                 get_peak(gradient_col_2[col_p:])+col_p ]
    peaks_row = [get_peak(gradient_row_2[:row_p], if_back=True),
                 get_peak(gradient_row_2[row_p:])+row_p ]

    # if output_pic:
    #     axes[0][0].scatter(peaks_col, cols[peaks_col], marker='x', color='red')
    #     axes[0][1].scatter(peaks_row, rows[peaks_row], marker='x', color='red')
    #     axes[2][0].axvline(x=col_p, color='red')
    #     axes[2][1].axvline(x=row_p, color='red')
    #
    #     axes[2][0].scatter(peaks_col, gradient_col_2[peaks_col], marker='x', color='red')
    #     axes[2][1].scatter(peaks_row, gradient_row_2[peaks_row], marker='x', color='red')
    #
    #     plt.tight_layout()
    #     plt.savefig(storing_path + time + side + '_plot.png', dpi=500)
    #     plt.close()

    col_r = abs(peaks_col[1] - peaks_col[0])/2
    row_r = abs(peaks_row[1] - peaks_row[0])/2

    col_valid = valid_radius(col_r, iris_r)
    # print('valid col', col_valid, col_r)
    row_valid = valid_radius(row_r, iris_r)
    # print('valid row', row_valid, row_r)

    # print('diff', abs(col_r - row_r),  abs(col_r - row_r)//1 <= 10)
    if col_valid and row_valid:
        if abs(col_r - row_r)//1 <= 10 or abs(col_r - approx_r)//1 <= 5 or abs(row_r - approx_r)//1 <= 5:
            True, None, (None, None) # no need to adjust the radius
        else:
            if abs(approx_r - row_r) <= abs(approx_r - col_r):
                return False, row_r, (None, row_p)
            else: return False, col_r, (col_p, None)

    elif not col_valid and not row_valid: return True, None, (None, None) # cannot adjust the radius

    elif row_valid: return False, row_r, (None, row_p)
    elif col_valid: return False, col_r, (col_p, None)

    return False, col_r, (None, None)

def smooth_row(img, approx_r):
    row, col = img.shape
    total = img.sum(axis=1)
    total = total/255
    zero_row = []
    for i in range(row-1):
        if total[i] == 0 and total[i+1] != 0:
            zero_row.append(i+1)
            break

    for i in reversed(range(row)):
        if total[i] == 0 and total[i-1] != 0:
            zero_row.append(i)
            break

    for a in range(zero_row[0], zero_row[1]):
        if total[a] < approx_r-15:
            img[a] = np.full(col, 0)

    # print(row/6//1, ' row smoothing')
    for a in range( int(row/6//1) ):
        a = a+zero_row[0]
        left = total[a]
        right = total[a*(-1)-1]

        if left == 0 and right != 0:
            img[a*(-1)-1] = np.full(col, 0)
        elif left != 0 and right == 0:
            img[a] = np.full(col, 0)

    return img


def smooth_img(img, approx_r):
    row, col = img.shape
    total = img.sum(axis=0)
    total = total/255
    zero_row = []
    for i in range(col-1):
        if total[i] == 0 and total[i+1] != 0:
            zero_row.append(i+1)
            break

    for i in reversed(range(col)):
        if total[i] == 0 and total[i-1] != 0:
            zero_row.append(i)
            break

    if len(zero_row) != 2: return img

    for a in range(zero_row[0], zero_row[1]):
        if total[a] < approx_r-15:
            img[:, a] = np.full(row, 0)

    # print(col/10//1, ' col smoothing')
    for a in range( int(col/10//1) ):
        a = a+zero_row[0]
        left = total[a]
        right = total[a*(-1)-1]

        if abs(left - right) >= approx_r-10:
            if left == 0 and right != 0:
                img[:, a*(-1)-1] = np.full(row, 0)
            elif left != 0 and right == 0:
                img[:, a] = np.full(row, 0)
    return img

def get_rects(file):
    frame = cv2.imread(file)
    frame = imutils.resize(frame, width=1200, height=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)
    return rects, gray, frame

def extract_eye(frame, shape, start, end):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[start:end]]))
    roi = frame[y:(y + h), x:(x + w)]
    roi = imutils.resize(roi, width=100*3, height=75*3, inter=cv2.INTER_CUBIC)
    roi = convert_gray(roi)
    return roi


def get_eye(frame, shape):
    # shape = predictor(gray, rect)
    # shape = imutils.face_utils.shape_to_np(shape)
    left_eye = extract_eye(frame, shape, lStart, lEnd)
    left_eye = resize(left_eye)
    right_eye = extract_eye(frame, shape, rStart, rEnd)
    right_eye = resize(right_eye)
    return left_eye, right_eye


def convert_gray(gray):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (5, 5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (5, 5))
    gray = cv2.GaussianBlur(gray, (7, 7), 3)
    # gray = cv2.fastNlMeansDenoising(gray, gray, h=20)
    return gray


def get_binary(gray, t, side=None, time=None, output_pic=False, storing_path=None):
    _, thres = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    blur = cv2.medianBlur(open, 5)
    # blur = smooth_row(blur, 27)
    # blur = smooth_img(blur, 27)
    # print(t)
    # cv2.imshow('binary', blur)
    if output_pic:
        cv2.imwrite(storing_path + time + side + '_combo.jpg', np.hstack([gray, blur]))

    return blur
    # edge = cv2.Canny(gray, t-10, t+20)
    # edge = cv2.dilate(edge, (3, 3), 3)
    # edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)

    # kernel = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    # emboss = cv2.filter2D(gray, -1, kernel) + 128

    # bor = cv2.bitwise_or(edge, blur)
    # band = cv2.bitwise_and(emboss, blur)
    # band = cv2.dilate(band, (3, 3), 3)
    # band = cv2.medianBlur(band, 3)

    # if output_pic:
        # cv2.imwrite(storing_path + time + side + '_combo.jpg', np.hstack([gray, blur, edge, bor, band, emboss]))

        # image = imutils.resize(gray, width=100 * 10, height=75 * 10, inter=cv2.INTER_CUBIC)
        # row, col = image.shape
        # for a in range(0, row, 10):
        #     for b in range(0, col, 10):
        #         if image[a][b] <= t:
        #             image = cv2.putText(image, str(image[a][b]), (b, a), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        # cv2.imwrite(storing_path + time + side + 'eye.jpg', image)
    # return blur, band, edge


def find_closest(cnts, approx_x, approx_y, previous_pupil):
    #compare with previous pupil
    if len(cnts) == 0:
        return None

    distance_ary = []
    for i in cnts:
        (x, y), r = cv2.minEnclosingCircle(i)
        if r < 5: continue
        d_approx = np.sqrt(np.square(approx_x - x) + np.square(approx_y - y))
        if previous_pupil is not None:
            comp_x, comp_y, comp_r, = previous_pupil
            d_prev = np.sqrt(np.square(comp_x - x) + np.square(comp_y - y))
        else:
            d_prev = 0
        distance_ary.append((d_approx, d_prev, [x, y, r]))

    if len(distance_ary) == 0: return None

    distance_ary = sorted(distance_ary, key=lambda x: (x[0], x[1]))
    # print('[DETECTED CIRCLES TO APPROX CENTER] ------------- ')
    # print('[APPROX CENTER]', approx_x, approx_y)

    return distance_ary[0][2]

def euclidean_dist(pnts, pnts_2):
    pnts = np.array(pnts)
    pnts_2 = np.array(pnts_2)
    d = np.sum(np.square(pnts - pnts_2) )
    return np.sqrt(d)

def select_closest(cnts, approx_x, approx_y, approx_r):
    if cnts is None or len(cnts) == 0: return None

    ary = []
    for c in cnts:
        (x, y), r = c
        if r <= LOWER_LIMIT or r >= UPPER_LIMIT: continue #+np.square(r-approx_r)
        ary.append(( euclidean_dist( [x, y, r], [approx_x, approx_y, approx_r]), c ))

    if len(ary) == 0: return None

    ary = sorted(ary, key=lambda x: x[0])
    return ary[0][-1]

# def clean_img(gray, ary):
#     if ary is None: return gray
#     start, end = ary
#     row, col = gray.shape
#     for a in range(row):
#         for b in range(col):
#             if b < start-5 or b > end+5: gray[a][b] = 255
#     return gray

def clean_img(gray, x, y, r):
    row, col = gray.shape
    rang = (x-r-5, x+r+5)
    for a in range(row):
        for b in range(col):
            if b < rang[0] or b > rang[1]: gray[a][b] = 255

    for a in range(5):
        gray[a] = np.full(col, 255)
        gray[a*(-1)-1] = np.full(col, 255)

    return gray


def cht(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    # circles = np.uint16(np.around(circles))
    if circles is not None:
        # circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
        # print(circles)
        circles = [ ((a[0], a[1]), a[2]) for a in circles[0]]
    return circles





