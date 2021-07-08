import numpy as np

def get_threshold(gray, range_x=15, range_r=0):
    start = np.around(range_x-range_r, decimals=0)
    end = np.around(range_x+range_r, decimals=0)
    cols = []
    row, col = gray.shape

    if range_x == 15:
        end = col-15
    print(start, ' -> ', end)

    for a in np.arange(start, end):
        a = int(a)
        sum = []
        for b in range(row):
            if gray[b][a] != 0 or gray[b][a] != 255:
                sum.append(gray[b][a])

        cols.append(np.mean(sum))

    return np.around(max(cols), decimals=0), np.around(min(cols), decimals=0)


