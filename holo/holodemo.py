import os
import glob
from shutil import copyfile
from holovideo import make_video


def run_imitator(img_dir, img_names, tgt_path, load_path, output_dir):
	preds_name_mask = 'imitators/pred_*.jpg' 

	for img_name in img_names:
		subject_name = img_name.split('.')[0]
		src_path = os.path.join(img_dir,img_name)
		print ('Processing', subject_name)

		#Run imitator:
		os.system("python3 -W ignore run_imitator.py --gpu_ids 0 --model imitator --output_dir %s --src_path %s --tgt_path %s --has_detector --post_tune --save_res --load_path %s" % (output_dir, src_path, tgt_path, load_path))

		#Copy predicted images:
		src_preds_paths = glob.glob(os.path.join(output_dir,preds_name_mask))
		dst_preds_dir = os.path.join(output_dir,'preds_imitator',subject_name)

		if not os.path.exists(dst_preds_dir):
			os.makedirs(dst_preds_dir)

		for src_p in src_preds_paths:
			filename = src_p.split('/')[-1]
			dst_p = os.path.join(dst_preds_dir,filename)
			print (src_p)
			copyfile(src_p, dst_p)

	return True


def run_imitator_front_warp(img_dir, img_names, tgt_path, load_path, output_dir):
	preds_name_mask = 'imitators/pred_*.jpg' 

	for img_name in img_names:
		subject_name = img_name.split('.')[0]
		src_path = os.path.join(img_dir,img_name)
		print ('Processing', subject_name)

		#Run imitator:
		os.system("python3 -W ignore run_imitator.py --gpu_ids 0 --model imitator --output_dir %s --src_path %s --tgt_path %s --has_detector --post_tune --save_res --front_warp --load_path %s" % (output_dir, src_path, tgt_path,load_path))

		#Copy predicted images:
		src_preds_paths = glob.glob(os.path.join(output_dir,preds_name_mask))
		dst_preds_dir = os.path.join(output_dir,'preds_imitator_front_warp',subject_name)

		if not os.path.exists(dst_preds_dir):
			os.makedirs(dst_preds_dir)

		for src_p in src_preds_paths:
			filename = src_p.split('/')[-1]
			dst_p = os.path.join(dst_preds_dir,filename)
			print (src_p)
			copyfile(src_p, dst_p)

	return True


def run_view(img_dir,img_names,load_path,output_dir):
	preds_name_mask = 'imgs/pred_*.jpg' 

	for img_name in img_names:
		subject_name = img_name.split('.')[0]
		src_path = os.path.join(img_dir,img_name)
		print ('Processing', subject_name)

		#Run imitator:
		os.system("python3 -W ignore run_view.py --gpu_ids 0 --model viewer --output_dir %s --src_path %s --bg_ks 13  --ft_ks 3 --has_detector --post_tune --save_res --bg_replace --load_path %s" % (output_dir, src_path,load_path))

		#Copy predicted images:
		src_preds_paths = glob.glob(os.path.join(output_dir,preds_name_mask))
		dst_preds_dir = os.path.join(output_dir,'preds_view',subject_name)

		if not os.path.exists(dst_preds_dir):
			os.makedirs(dst_preds_dir)

		for src_p in src_preds_paths:
			filename = src_p.split('/')[-1]
			dst_p = os.path.join(dst_preds_dir,filename)
			print (src_p)
			copyfile(src_p, dst_p)

	return True


def run_view_front_warp(img_dir,img_names,load_path,output_dir):
	preds_name_mask = 'imgs/pred_*.jpg' 

	for img_name in img_names:
		subject_name = img_name.split('.')[0]
		src_path = os.path.join(img_dir,img_name)
		print ('Processing', subject_name)

		#Run imitator:
		os.system("python3 -W ignore run_view.py --gpu_ids 0 --model viewer --output_dir %s --src_path %s --bg_ks 13  --ft_ks 3 --has_detector --post_tune --save_res --bg_replace --front_warp --load_path %s" % (output_dir, src_path,load_path))

		#Copy predicted images:
		src_preds_paths = glob.glob(os.path.join(output_dir,preds_name_mask))
		dst_preds_dir = os.path.join(output_dir,'preds_view_front_warp',subject_name)

		if not os.path.exists(dst_preds_dir):
			os.makedirs(dst_preds_dir)

		for src_p in src_preds_paths:
			filename = src_p.split('/')[-1]
			dst_p = os.path.join(dst_preds_dir,filename)
			print (src_p)
			copyfile(src_p, dst_p)

	return True


def main():
	img_dir = '/home/darkalert/KazendiJob/Data/LWGtest/'
	img_names = []
	img_names.append('inet-woman1.jpeg')
	img_names.append('inet-woman2.jpeg')
	img_names.append('inet-woman3.jpeg')
	img_names.append('inet-man1.jpeg')
	img_names.append('inet-man3.jpeg')
	img_names.append('jason.jpeg')
	tgt_path = './assets/samples/refs/iPER/024_8_2'
	load_path = './outputs/checkpoints/my_iPER/net_epoch_26_id_G.pth'
	output_dir = './outputs/results/LWGtest_by_my_model/'

	os.chdir("../")

	#Run impersonator:
	run_imitator(img_dir,img_names,tgt_path,load_path,output_dir)
	run_imitator_front_warp(img_dir,img_names,tgt_path,load_path,output_dir)
	run_view(img_dir,img_names,load_path,output_dir)
	run_view_front_warp(img_dir,img_names,load_path,output_dir)

	#Make videos:
	seq_dir = os.path.abspath(output_dir)
	seq_names = ['preds_imitator','preds_imitator_front_warp']

	subj_names = [img_names[0],img_names[1],img_names[2]]
	output_mp4_path = os.path.join(seq_dir,'imitator-women.mp4')
	make_video(subj_names, img_dir, seq_names, seq_dir, output_mp4_path)

	subj_names = [img_names[3],img_names[4],img_names[5]]
	output_mp4_path = os.path.join(seq_dir,'imitator-men.mp4')
	make_video(subj_names, img_dir, seq_names, seq_dir, output_mp4_path)

	seq_names = ['preds_view','preds_view_front_warp']

	subj_names = [img_names[0],img_names[1],img_names[2]]
	output_mp4_path = os.path.join(seq_dir,'view-women.mp4')
	make_video(subj_names, img_dir, seq_names, seq_dir, output_mp4_path, fps=5)

	subj_names = [img_names[3],img_names[4],img_names[5]]
	output_mp4_path = os.path.join(seq_dir,'view-men.mp4')
	make_video(subj_names, img_dir, seq_names, seq_dir, output_mp4_path, fps=5)


if __name__ == "__main__":
	main()