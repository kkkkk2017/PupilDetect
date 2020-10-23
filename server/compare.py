# from eye_detect import Eye_Detector
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter

def input_data(file_1, file_2):
    df = pandas.read_csv(file_1)

    if file_2.__contains__('.csv'):
        tdf = pandas.read_csv(file_2)
        tobii_left = tdf['left_pupil']
        tobii_right = tdf['right_pupil']
        # tobii_mean = tdf['mean']
    else:
        tdf = pandas.read_excel(file_2, index_col=0, sheet_name='Data')
        tobii_left = tdf['Pupil diameter left [mm]'][:len(df['left_pupil'])]
        tobii_right = tdf['Pupil diameter right [mm]'][:len(df['left_pupil'])]
        # tobii_mean = []
        # for i, c in zip(tobii_left, tobii_right):
        #     tobii_mean.append(np.mean((i, c)))

    return df, tobii_left, tobii_right, None

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

def cal_PLR(pupil, iris):
    if pupil != 0 and iris != 0:
        return (pupil*2)/(iris*2)
    return 0

def preprocess(df):
    blink_count = df['blink_count']
    blink_count = blink_count.fillna(0)
    df['blink_count'] = blink_count

    df = df.dropna()
    print('[RAW]total values:', len(df['left_pupil']))

    total_blink = np.sum([i for i in df['blink_count']])
    print('total blink', total_blink)

    left_list = {}
    right_list = {}

    left_PLR_list = {}
    right_PLR_list = {}
    start_t = float(df.iloc[0]['time'])
    end_t = float(df.iloc[-1]['time'])
    total_sec = end_t - start_t

    print('total duration:', total_sec, 'sec')
    total_index = int(np.around([total_sec])[0])

    for (_, d1), (_, d2) in zip(df[0:-2].iterrows(), df[1:-1].iterrows()):
        time1 = float(d1['time']) - start_t
        left1 = float(d1['left_pupil'])
        left_iris1 = float(d1['left_iris'])
        right1 = float(d1['right_pupil'])
        right_iris1 = float(d1['right_iris'])

        left_PLR_1 = cal_PLR(left1, left_iris1)
        right_PLR_1 = cal_PLR(right1, right_iris1)

        time2 = float(d2['time']) - start_t
        left2 = float(d2['left_pupil'])
        left_iris2 = float(d1['left_iris'])
        right2 = float(d2['right_pupil'])
        right_iris2 = float(d1['right_iris'])

        left_PLR_2 = cal_PLR(left2, left_iris2)
        right_PLR_2 = cal_PLR(right2, right_iris2)

        # within 1 sec
        if (time2 - time1) < 1:
            left_list.update(add_num(time1//1, left_PLR_1, left_PLR_2))
            right_list.update(add_num(time1//1, right_PLR_1, right_PLR_2))
        else:
            left_PLR_list.update({time1//1: left_PLR_1})
            right_PLR_list.update({time1//1: right_PLR_1})
            left_list.update({time1//1: left1})
            right_list.update({time1//1: right1})

    return left_list, right_list, left_PLR_list, right_PLR_list

def add_num(time, num1, num2):
    if num1 == 0 and num2 != 0:
        return {time: num2}
    elif num2 == 0 and num1 != 0:
        return {time: num1}
    elif num1 != 0 and num2 != 0:
        num = np.average([num1, num2])
        return {time: num}
    else:
        return {time: 0}

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
            if np.abs(percent) <= 0.1:
                deltas.update({i: percent})
        # else:
        #     deltas.update({i: 0})

    average_changes = np.average([np.abs(i) for i in deltas.values() if i != 0])
    # print('data: ', arr)
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
    print('non-0 values:', len(x))
    return x, y

def cal_amplitude(d, sec=2):
    result = []
    if isinstance(d, dict):
        for i in range(0, len(d.keys()), sec):
            v1 = d.get(i)
            v2 = d.get(i+2)
            if v2 is None:
                break
            result.append((v2-v1)*0.1*0.5)
    elif isinstance(d, list):
        for i in range(0, len(d), sec):
            v1 = d[i]
            if i+2 >= len(d):
                break
            v2 = d[i+2]
            result.append(v2-v1)

    return result

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='filenames')
    my_parser.add_argument('file_1',
                           help='csv file')
    my_parser.add_argument('file_2',
                           help='csv/xlsx file')

    args = my_parser.parse_args()
    file_1 = args.file_1
    file_2 = args.file_2

    df, tobii_left, tobii_right, tobii_mean = input_data(file_1, file_2)
    left_pupil, right_pupil, left_PLR, right_PLR = preprocess(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # left_PLR_x, left_PLR_y = sort_dict(left_PLR)
    # right_PLR_x, right_PLR_y = sort_dict(right_PLR)
    # mean_x, mean_y = sort_dict(comp_mean)

    # ax1.plot(np.zeros(max(left_x)), color='red', label='y=0')
    # ax2.plot(np.zeros(max(right_x)), color='red', label='y=0')
    # ax3.plot(np.zeros(max(mean_x)), color='red', label='y=0')

    #draw PLR
    # ax1.plot(left_x, left_y, color='green', label='opencv program left PLR')
    # ax2.plot(right_x, right_y, color='blue', label='opencv program right PLR')
    # # ax3.plot(mean_x, mean_y, color='pink', label='file 1 mean')
    #
    # ax1.scatter(comp_left.keys(), comp_left.values(), color='red', s=2)
    # ax2.scatter(comp_right.keys(), comp_right.values(), color='red', s=2)
    # ax3.scatter(comp_mean.keys(), comp_mean.values(), color='purple', s=1)

    # print('\ncalculating {} left'.format(file_2))
    # tobii_left = cal_delta2(tobii_left[10:-10])
    # print('calculating {} right'.format(file_2))
    # tobii_right = cal_delta2(tobii_right[10:-10])

    left_x, left_y = sort_dict(left_pupil)
    right_x, right_y = sort_dict(right_pupil)

    # print('\ncalculating {} left'.format(file_1))
    # left = cal_delta2(df['left_pupil'])
    # print('calculating {} right'.format(file_1))
    # right = cal_delta2(df['right_pupil'])
    left = cal_amplitude(left_pupil)
    right = cal_amplitude(right_pupil)
    width=0.35

    ax1.bar(np.arange(len(left))+width, left, color='yellow', label='{} left_pupil_dilation'.format(file_1))
    ax2.bar(np.arange(len(right))+width, right, color='salmon', label='{} right_pupil_dilation'.format(file_1))

    # ploting file 2

    tobii_left = cal_amplitude(list(tobii_left.values))[:len(left)]
    tobii_right = cal_amplitude(list(tobii_right.values))[:len(right)]

    # tobii_left_x = list(tobii_left.keys())[: list(left.keys())[-1]]
    # tobii_right_x = list(tobii_right.keys())[: list(right.keys())[-1]]
    #
    # tobii_left_y = list(tobii_left.values())[: list(left.keys())[-1]]
    # tobii_right_y = list(tobii_right.values())[: list(right.keys())[-1]]

    ax1.bar(np.arange(len(tobii_left))+width, tobii_left, width, color='lightblue', label='{} left_pupil_dilation'.format(file_2))
    ax2.bar(np.arange(len(tobii_right))+width, tobii_right, width, color='pink', label='{} right_pupil_dilation'.format(file_2))

    # ax1.scatter(tobii_left_x, tobii_left_y, color='purple', s=2)
    # ax2.scatter(tobii_right_x, tobii_right_y, color='purple', s=2)

    # ax1.scatter(left.keys(), left.values(), color='purple', s=2)
    # ax2.scatter(right.keys(), right.values(), color='purple', s=2)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    # ax3.legend(loc='upper right')

    ax1.grid()
    ax2.grid()

    # ax1.set_xlim(right=0)
    ax2.set_xlim(left=0)

    # ax3.grid()
    x_ticks = np.arange(0, len(left), 10)
    plt.xticks(x_ticks)

    plt.xlabel('2 seconds')
    plt.ylabel('dilation rate')
    plt.show()