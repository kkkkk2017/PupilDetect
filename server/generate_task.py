import random

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'X', 'Y', 'Z']
#'M', 'W',

file_name={
    0: '0-back',
    1: '1-back',
    2: '2-back',
    3: '3-back',
}

if __name__ == '__main__':
    for i in range(4):
        file = 'tasks/'+file_name.get(i)+'.txt'
        with open(file, 'a') as f:
            total = int(3*60/2)

            for c in range(total):
                if i < 2:
                    num = random.randint(0, 14)
                else:
                    num = random.randint(0, 10)
                let = letters[num]
                f.write(let+'\n')

            f.close()
