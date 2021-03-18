file = '1-back.txt'

if __name__ == '__main__':
    all = 0
    with open('1-back.txt', 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines[1:]):
            if lines[i] == lines[i-1]:
                all += 1
    print(all)
    print(len(lines))

    all = 0
    with open('2-back.txt', 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines[2:]):
            if lines[i] == lines[i-2]:
                all += 1
    print(all)
    print(len(lines))


    all = 0
    with open('3-back.txt', 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines[3:]):
            if lines[i] == lines[i-3]:
                all += 1
    print(all)
    print(len(lines))