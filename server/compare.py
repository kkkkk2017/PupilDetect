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
        tobii_mean = tdf['mean']
    else:
        tdf = pandas.read_excel(file_2, index_col=0, sheet_name='Data')
        tobii_left = tdf['Pupil diameter left [mm]'][:len(df['left_pupil'])]
        tobii_right = tdf['Pupil diameter right [mm]'][:len(df['left_pupil'])]
        tobii_mean = []
        for i, c in zip(tobii_left, tobii_right):
            tobii_mean.append(np.mean((i, c)))

    return df, tobii_left, tobii_right, tobii_mean

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

def cal_delta(arr):
    arr = [i for i in arr if not np.isnan(i)]
    arr = filter(arr)
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

    ax1.plot(left, color='black', label='file 2 left pupil')
    ax2.plot(right, color='black', label='file 2 right pupil')
    ax3.plot(mean, color='black', label='file 2 mean')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='filenames')
    my_parser.add_argument('file_1',
                           help='csv file')
    my_parser.add_argument('file_2',
                           help='csv/xlsx file')

    args = my_parser.parse_args()
    df, tobii_left, tobii_right, tobii_mean = input_data(args.file_1, args.file_2)
    draw_precentage(df, tobii_left, tobii_right, tobii_mean)