import os
import shutil

#step 4: assign labels and create folders for each "film"
### video_data -> saved_frame -> remove_lastimgs-> stress_data_s8

#The labeling rule is different from the suggestion in the forum. For example, the "TCS-02_07_12-1130-M.avi"
#should be 'calm,calm,calm,stressful,stressful,stressful' if using the labeling rule in forum, while
# it should be 'stressful,stressful,stressful,calm,calm,calm' if you manuly check it.
# So, I mannuly check a lot of videos and finally find the really correct labeling rules.


labels2=["Stressful", "Stressful","Stressful" ,"Calm", "Calm","Calm"]
labels1=["Calm", "Calm", "Calm", "Stressful","Stressful","Stressful"]

# print(len(labels2))
root_dir='../saved_frame/'
videos=sorted(os.listdir(root_dir))
print(len(videos))
print(videos)
video_count=1
dest_path=''
print(videos)
for video in range(len(videos)):
        #print('video-=-----',video)
        filenames = sorted(os.listdir(os.path.join(root_dir, videos[video])))
        print('f',filenames)#len:560
        film_count = 0
        for i in range(len(filenames)):
                if i>=28*6:
                        break
                if i%28==0:
                        if videos[video][:3]=='TSC':
                                suffix_name = 'S' + str(video + 1) + '_' + str(labels1[film_count]) + '_F' + str(film_count + 1)
                        else:
                                suffix_name = 'S' + str(video + 1) + '_' + str(labels2[film_count]) + '_F' + str(film_count + 1)
                        dest_path='../stress_data_s7/' + suffix_name
                        os.makedirs(dest_path)
                        film_count += 1
                file_path=filenames[i]
                origin_path=root_dir+videos[video]+'/'+file_path
                shutil.move(origin_path,dest_path)






