# from eye_detect import Eye_Detector
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter

def input_data(file_1, file_2):
    df = pandas.read_csv(file_1)

    # if file_2.__contains__('.csv'):
    #     tdf = pandas.read_csv(file_2)
    #     tobii_left = tdf['left_pupil']
    #     tobii_right = tdf['right_pupil']
    #     tobii_mean = tdf['mean']
    # else:
    #     tdf = pandas.read_excel(file_2, index_col=0, sheet_name='Data')
    #     tobii_left = tdf['Pupil diameter left [mm]'][:len(df['left_pupil'])]
    #     tobii_right = tdf['Pupil diameter right [mm]'][:len(df['left_pupil'])]
    #     tobii_mean = []
    #     for i, c in zip(tobii_left, tobii_right):
    #         tobii_mean.append(np.mean((i, c)))

    return df, [], [], []

def draw_raw_data(df, tobii_left, tobii_right, tobii_mean):
    # print(tdf.columns)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    x=[i for i in range(len(df['left_pupil']))]
    df.plot(kind='line', ax=ax1, y='left_pupil', figsize=(10,10), color='black')
    df.plot(kind='line', ax=ax2, y='right_pupil', figsize=(10,10), color='black')
    df.plot(kind='line', ax=ax3, y='mean', figsize=(10,10), color='black')

    ax1.plot([i for i in range(len(tobii_left))], tobii_left, color='green', label='tobii left pupil')
    ax2.plot([i for i in range(len(tobii_right))], tobii_right, color='blue', label='tobii right pupil')
    ax3.plot(tobii_mean, color='red', label='tobii mean')
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()

def filter(arr):
    most_common = Counter([i for i in arr]).most_common()[0][0]
    output = [i for i in arr if np.abs(i - most_common) <= 1]
    print('most=', most_common)
    return output

def read_input(df):
    total_blink = np.sum([i for i in df['blink_count'] if not np.isnan(i)])
    print(df.head())
    left_list = {}
    right_list = {}
    start_t = float(df.iloc[0]['time'])
    end_t = float(df.iloc[-1]['time'])
    total_sec = end_t - start_t
    total_index = int(np.around([total_sec])[0])

    for (_, d1), (_, d2) in zip(df[0:-2].iterrows(), df[1:-1].iterrows()):
        time1 = float(d1['time']) - start_t
        left1 = float(d1['left_pupil'])
        right1 = float(d1['right_pupil'])

        time2 = float(d2['time']) - start_t
        left2 = float(d2['left_pupil'])
        right2 = float(d2['right_pupil'])

        # within 1 sec
        if (time2 - time1) < 1:
            left_list = add_num(left_list, time1//1, left1, left2)
            right_list = add_num(right_list, time1//1, right1, right2)
        else:
            left_list.update({time1//1: left1})
            right_list.update({time1//1: right1})

    left = []
    right = []
    mean = []
    # for i in range(total_index):
    #     num = left_list.get(i)
    #     if num:
    #         left.append(num)
    #     else:
    #         left.append(0)

        # num2 = right_list.get(i)
        # if num2:
        #     right.append(num2)
        # else:
        #     right.append(0)

        # if left[i] != 0 and right[i] != 0:
        #     mean.append(np.mean((left[i], right[i])))
        # elif left[i] == 0 and right[i] != 0:
        #     mean.append(right[i])
        # elif left[i] != 0 and right[i] == 0:
        #     mean.append(left[i])
        # else:
        #     mean.append(0)

    return left_list, right_list, mean

def add_num(arr, time, num1, num2):
    if num1 == 0 and num2 != 0:
        arr.update({time: num2})
    elif num2 == 0 and num1 != 0:
        arr.update({time: num1})
    elif num1 != 0 and num2 != 0:
        num = np.average([num1, num2])
        arr.update({time: num})
    else:
        arr.update({time: 0})
    return arr

def cal_delta(arr):
    arr = [i for i in arr if not np.isnan(i)]
    # arr = filter(arr)
    deltas = []
    for i in range(1, len(arr)-1):
        if arr[i] != 0 and arr[i-1] != 0:
            delta = arr[i] - arr[i-1]
            percent = delta/arr[i-1]
            deltas.append(percent)
        else:
            deltas.append(0)

    average_changes = np.average([np.abs(i) for i in deltas])
    print('data: ', arr)
    print('average changes', average_changes)
    return deltas

def draw_precentage(df, tobii_left, tobii_right, tobii_mean):
    print('open_cv left: ')
    comp_left = cal_delta(df['left_pupil'])
    print('open_cv right: ')
    comp_right = cal_delta(df['right_pupil'])
    print('open_cv mean: ')
    comp_mean = cal_delta(df['mean'])

    print('eye_tracker left: ')
    left = cal_delta(tobii_left)
    print('eye_tracker right: ')
    right = cal_delta(tobii_right)
    print('eye_tracker mean: ')
    mean = cal_delta(tobii_mean)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_yticks(np.arange(min(min(comp_left), min(left)), max(max(comp_left), max(left)), step=0.1))
    ax2.set_yticks(np.arange(min(min(comp_right), min(right)), max(max(comp_right), max(right)), step=0.1))
    ax3.set_yticks(np.arange(min(min(comp_mean), min(mean)), max(max(comp_mean), max(mean)), step=0.05))

    x_1 = [0 for _ in range(len(comp_left))]
    x_2 = [0 for _ in range(len(comp_right))]
    x_3 = [0 for _ in range(len(comp_mean))]
    ax1.plot(x_1, color='red', label='y=0')
    ax2.plot(x_2, color='red', label='y=0')
    ax3.plot(x_3, color='red', label='y=0')

    ax1.plot(comp_left, color='green', label='file 1 left_pupil_dilation')
    ax2.plot(comp_right, color='blue', label='file 1 right_pupil_dilation')
    ax3.plot(comp_mean, color='pink', label='file 1 mean')

    # ax1.plot(left, color='black', label='file 2 left pupil')
    # ax2.plot(right, color='black', label='file 2 right pupil')
    # ax3.plot(mean, color='black', label='file 2 mean')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.show()

def cal_delta2(arr):
    arr = [i for i in arr if not np.isnan(i)]
    deltas = {}
    for i in range(1, len(arr)-1):
        if arr[i] != 0 and arr[i-1] != 0:
            delta = arr[i] - arr[i-1]
            percent = delta/arr[i-1]
            deltas.update({i: percent})
        # else:
        #     deltas.update({i: 0})

    average_changes = np.average([np.abs(i) for i in deltas.values() if i != 0])
    print('data: ', arr)
    print('average changes', average_changes)
    return deltas

def sort_dict(dict):
    x = []
    y = []
    for i in sorted(dict.keys()):
        key = i
        value = dict.get(key)
        if value != 0:
            x.append(i)
            y.append(value)
    print(len(x))
    return x, y


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='filenames')
    my_parser.add_argument('file_1',
                           help='csv file')
    my_parser.add_argument('file_2',
                           help='csv/xlsx file')

    args = my_parser.parse_args()
    df, tobii_left, tobii_right, tobii_mean = input_data(args.file_1, args.file_2)
    left_pupil, right_pupil, mean_pupil = read_input(df)

    # print('open_cv left: ')
    # comp_left = cal_delta2(left_pupil)
    # print('open_cv right: ')
    # comp_right = cal_delta2(right_pupil)
    # print('open_cv mean: ')
    # comp_mean = cal_delta2(mean_pupil)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    comp_left = left_pupil
    comp_right = right_pupil
    comp_mean = mean_pupil

    x_1 = comp_left.keys()
    x_2 = comp_right.keys()
    # x_3 = comp_mean.keys()

    left_x, left_y = sort_dict(comp_left)
    right_x, right_y = sort_dict(comp_right)
    # mean_x, mean_y = sort_dict(comp_mean)

    # ax1.plot(np.zeros(max(left_x)), color='red', label='y=0')
    # ax2.plot(np.zeros(max(right_x)), color='red', label='y=0')
    # ax3.plot(np.zeros(max(mean_x)), color='red', label='y=0')

    ax1.plot(left_x, left_y, color='green', label='file 1 left_pupil_dilation')
    ax2.plot(right_x, right_y, color='blue', label='file 1 right_pupil_dilation')
    # ax3.plot(mean_x, mean_y, color='pink', label='file 1 mean')

    ax1.scatter(comp_left.keys(), comp_left.values(), color='purple', s=2)
    ax2.scatter(comp_right.keys(), comp_right.values(), color='purple', s=2)
    # ax3.scatter(comp_mean.keys(), comp_mean.values(), color='purple', s=1)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    # ax3.legend(loc='upper right')

    ax1.grid()
    ax2.grid()
    # ax3.grid()
    plt.show()