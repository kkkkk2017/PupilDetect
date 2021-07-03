import cv2
from os import listdir, path
from os.path import isfile, join
import img_utility
import numpy as np

participant = 'light_eye'
storing_path = path.expanduser('~/Desktop/test_eye_images/' + participant + '/')
HSV = []
HSV_H = []
HSV_S = []
GRAY = []

if __name__ == '__main__':

    for d in ['HSV', 'GRAY', 'HSV_S', 'HSV_H']:
        onlyfiles = [f for f in listdir(storing_path + d) if isfile(join(storing_path + d, f)) and (f.__contains__('edge') or f.__contains__('thres'))]
        # print(join(storing_path + d, onlyfiles[0]))
        # for f in onlyfiles:
        #     image = cv2.imread(join(storing_path + d, f), 0)
        #     edge = cv2.Canny(image, 60, 100)
        #     cv2.imwrite(storing_path+d+'/edge_' + f, edge)
        exec(d + '=onlyfiles')
        print(d, len(eval(d)))

    for f in HSV_H:
        image = cv2.imread(join(storing_path + 'GRAY', f), 0)
        otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(otsu_threshold)
        cv2.imwrite(storing_path + 'HSV_H' + '/thres_' + f, image_result)
    # combine edge pics of VSPACE AND HSV
    # for f in GRAY:
    #     print(f)
    #     if f not in HSV: continue
    #     if f not in HSV_H: continue
    #
    #     edge1 = cv2.imread(join(storing_path + 'GRAY', f), 0)
    #     edge2 = cv2.imread(join(storing_path + 'HSV', f), 0)
    #     edge3 = cv2.imread(join(storing_path + 'HSV_H', f), 0)
    #
    #
    #     band = cv2.add(edge1, edge2, edge3)
    #     cv2.imwrite(storing_path + '/band' + '/band_' + f, band)
        #
        # bor = cv2.bitwise_or(edge1, edge2)
        # cv2.imwrite(storing_path + '/bor' + '/bor_' + f, bor)
        #
        # bnot = cv2.bitwise_not(edge1, edge2)
        # cv2.imwrite(storing_path + '/not' + '/not_' + f, bnot)

        # xor = cv2.bitwise_xor(edge1, edge2)
        # cv2.imwrite(storing_path + '/xor' + '/xor_' + f, xor)

    # while True:
    #
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #
    #     for file in HSV_S:
    #         image = cv2.imread(file, 0)
    #         image = cv2.GaussianBlur(image, (5, 5), 0)
    #
    #         otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #         print("Obtained threshold: ", otsu_threshold)
    #         cv2.imshow(file, np.hstack((image, image_result)))
