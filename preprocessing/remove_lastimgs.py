import os

#step3: remove images so that each folder has the same number of imgs

root_dir='../saved_frame/'
#
dir_names = sorted(list(os.listdir(root_dir)))
dir_names=sorted(dir_names)
print(dir_names)
for dir in range(len(dir_names)):
    filenames = sorted(os.listdir(os.path.join(root_dir, dir_names[dir])))
    print(len(filenames))
    print('f:',filenames)
    for i in range(len(filenames)):
        if i>=560:
            os.remove(os.path.join(root_dir, dir_names[dir], filenames[i]))
