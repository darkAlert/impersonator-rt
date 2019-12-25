import os
import glob
import shutil
import numpy as np
import cv2



def make_video(img_names, img_dir, seq_names, seq_dir, output_mp4_path, img_size = (256,256), fps = 24):
	rows = len(img_names)
	cols = len(seq_names)

	#Get frames paths:
	frames_paths = []
	for img_name in img_names:
		subj_name = img_name.split('.')[0]
		subj_paths = []
		for seq_name in seq_names:
			paths = glob.glob(os.path.join(seq_dir,seq_name,subj_name,'*.*'))
			paths.sort()
			subj_paths.append(paths)
		frames_paths.append(subj_paths)

	#Get min number of frames:
	min_frames = len(frames_paths[0][0])
	for i in range(rows):
		for j in range(cols):
			min_frames = min(min_frames,len(frames_paths[i][j]))
	assert min_frames > 0

	#Load subjects' images:
	subj_imgs = []
	for img in img_names:
		img = cv2.imread(os.path.join(img_dir,img),1)
		if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
			img = cv2.resize(img,img_size)
		subj_imgs.append(img)

	temp_dir = './outputs/results/temp_video'
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	else:
		shutil.rmtree(temp_dir)

	#Load and concatenate frames:
	for fi in range(min_frames):
		#Make frames:
		frame_imgs = []
		for ri in range(rows):
			row_imgs = []
			for ci in range(cols):
				img = cv2.imread(frames_paths[ri][ci][fi],1)
				if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
					img = cv2.resize(img,img_size)
				row_imgs.append(img)
			frame_imgs.append(np.concatenate([subj_imgs[ri]] + row_imgs, axis=1))
		canvas = np.concatenate(frame_imgs, axis=0)
		out_path = os.path.join(temp_dir,str('%05d' % fi) + '.png')
		cv2.imwrite(out_path,canvas)

	#Make a video:
	os.system("ffmpeg -f image2 -framerate %s -i %s/%%5d.png %s" % (fps,temp_dir,output_mp4_path))
	print ('Video has been saved to',output_mp4_path)

	shutil.rmtree(temp_dir)

	return True

