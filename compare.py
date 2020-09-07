# from eye_detect import Eye_Detector
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

def cal_delta(arr):
    arr = [i for i in arr if not np.isnan(i)]
    deltas = []
    for i in range(1, len(arr)-1):
        delta = arr[i] - arr[i-1]
        percent = delta/arr[i-1]
        deltas.append(percent)

    average_changes = np.average([np.abs(i) for i in deltas])
    print('average changes', average_changes)
    return deltas

def draw_precentage(df, tobii_left, tobii_right, tobii_mean):
    comp_left = cal_delta(df['left_pupil'])
    comp_right = cal_delta(df['right_pupil'])
    comp_mean = cal_delta(df['mean'])

    left = cal_delta(tobii_left)
    right = cal_delta(tobii_right)
    mean = cal_delta(tobii_mean)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(comp_left, color='green', label='file 1 left_pupil_dilation')
    ax2.plot(comp_right, color='blue', label='file 1 right_pupil_dilation')
    ax3.plot(comp_mean, color='red', label='file 1 mean')

    ax1.plot(left, color='black', label='file 2 left pupil')
    ax2.plot(right, color='black', label='file 2 right pupil')
    ax3.plot(mean, color='black', label='file 2 mean')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='filenames')
    my_parser.add_argument('file_1',
                           help='your file')
    my_parser.add_argument('file_2',
                           help='tobii file')

    args = my_parser.parse_args()
    df, tobii_left, tobii_right, tobii_mean = input_data(args.file_1, args.file_2)
    draw_precentage(df, tobii_left, tobii_right, tobii_mean)