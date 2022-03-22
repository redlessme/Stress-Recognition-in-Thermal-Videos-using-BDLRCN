import os

#step2: run image_processing.py, it will remove first and last 4 frames for each "film", the aim is to
# avoid wrongly dividing "flims".

root_dir='../saved_frame/'
#
dir_names = sorted(list(os.listdir(root_dir)))
dir_names=sorted(dir_names)
print('dirname',dir_names)
print(len(dir_names))
for dir in range (len(dir_names)):
    count=0
    removed_index=[0,1,2,3,35,34,33,32]
    # print(filenames)
    filenames=sorted(os.listdir(os.path.join(root_dir,dir_names[dir])))
    print(len(filenames))
    for i in range(len(filenames)):
        if count in removed_index:
            os.remove(os.path.join(root_dir,dir_names[dir],filenames[i]))
        count+=1
        if count == 36:
            count = 0




