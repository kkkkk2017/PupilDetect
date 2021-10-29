import os
from shutil import copyfile

from_path = 'D:\\eve_dataset\\eve_dataset'
to_path = 'C:\\Users\\kaixi\\Desktop\\eve data analysis\\all_db'

want_file_1 = 'opencv_video_result.csv'
want_file_2 = 'webcam_c.h5'
want_file_3 = 'webcam_c.timestamps.txt'

for dir in os.listdir(from_path):
    print(dir)

    current_from = os.path.join(from_path, dir)

    if not os.path.isdir(current_from): continue

    current_to = os.path.join(to_path, dir)
    #create new depo in destination folder
    if not os.path.isdir(current_to):
        os.mkdir(current_to)
    else: continue

    #open sub folder in dir
    for subdir in os.listdir(current_from):
        filenum = subdir.split('_')[0][4:]
        print(filenum)
        #copy files

        if not os.path.isfile(os.path.join(current_from, subdir, want_file_1)): continue

        copyfile(os.path.join(current_from, subdir, want_file_1), os.path.join(current_to, filenum+'_'+want_file_1))
        copyfile(os.path.join(current_from, subdir, want_file_2), os.path.join(current_to, filenum+'_'+want_file_2))
        copyfile(os.path.join(current_from, subdir, want_file_3), os.path.join(current_to, filenum+'_'+want_file_3))





