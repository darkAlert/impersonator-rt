import os
import glob
from tqdm import tqdm
import numpy as np
import cv2


# Replacing them as your own folder
dataset_video_root_path = '/home/darkalert/KazendiJob/Data/impersonator/iPER_1024_video_release'
save_images_root_path = '/home/darkalert/KazendiJob/Data/impersonator/iPER_1024_images'
target_size = (256,256)

def get_names(path_to_src, absolute_path = False):
    path_list = []
    for dirpath, dirnames, filenames in os.walk(path_to_src):
        for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
            if absolute_path:
                path_list.append(os.path.join(path_to_src,filename))
            else:
                path_list.append(filename)
    path_list.sort()

    return path_list


def extract_one_video(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.system("ffmpeg -i %s -start_number 0 %s/frame%%08d.png > /dev/null 2>&1" % (video_path, save_dir))


def main():
    global dataset_video_root_path, save_images_root_path

    video_path_list = sorted(glob.glob("%s/*.mp4" % dataset_video_root_path))

    for video_path in tqdm(video_path_list):
        video_name = os.path.split(video_path)[-1][:-4]
        actor_id, cloth_id, action_type = video_name.split('_')

        video_images_dir = os.path.join(save_images_root_path, actor_id, cloth_id, action_type)
        print ('Processing',video_images_dir)
        extract_one_video(video_path, video_images_dir)

        #Resize:
        paths = get_names(video_images_dir, True)
        for path in paths:
            img = cv2.imread(path, 1)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path,img)
        


        # import ipdb
        # ipdb.set_trace()


if __name__ == '__main__':
    main()

