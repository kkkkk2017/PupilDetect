import random

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

file_name={
    0: '0-back',
    1: '1-back',
    2: '1-back-speed',
    3: '2-back',
    4: '2-back-speed'
}

for i in range(5):
    file = 'tasks/'+file_name.get(i)+'.txt'
    with open(file, 'a') as f:
        if i == 2 or i == 4:
            total = 3*60
        else:
            total = 3*60/2

        for i in range(total):
            num = random.randint(0, 14)
            let = letters[num]
            f.write(let+'\n')
        f.close()
