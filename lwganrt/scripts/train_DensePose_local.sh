#! /bin/bash
# Train DensePose

# basic configs
gpu_ids=0

# dataset configs
dataset_mode=DensePose
root_dir=/home/darkalert/builds/ImpersonatorRT

# Holo dataset (for DensePose)
holo_data_dir=/home/darkalert/KazendiJob/Data/HoloVideo/Data  # need to be replaced!!!!!
holo_images_folder=avatars
holo_smpls_folder=smpls_by_vibe_aligned_lwgan
holo_uvs_folder=avatars_uv_256
holo_train_ids_file=train.txt
holo_test_ids_file=val.txt

# saving configs
checkpoints_dir=${root_dir}/lwganrt/outputs   # directory to save models, need to be replaced!!!!!
name=DensePose   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.

# model configs
model=densepose_trainer
gen_name=generator_uv
image_size=256

uv_mapping=${root_dir}/lwganrt/assets/pretrains/mapper.txt
hmr_model=${root_dir}/lwganrt/assets/pretrains/hmr_tf2pt.pth
smpl_model=${root_dir}/lwganrt/assets/pretrains/smpl_model.pkl
smpl_faces=${root_dir}/lwganrt/assets/pretrains/smpl_faces.npy
face_model=${root_dir}/lwganrt/assets/pretrains/sphere20a_20171020.pth
part_info=${root_dir}/lwganrt/assets/pretrains/smpl_part_info.json
front_info=${root_dir}/lwganrt/assets/pretrains/front_facial.json
head_info=${root_dir}/lwganrt/assets/pretrains/head.json

# pretrained G and D
load_G_path=/home/darkalert/builds/ImpersonatorRT/lwganrt/outputs/Holo_iPER/net_epoch_20_id_G.pth
n_threads_train=6

# training configs
batch_size=10
lambda_rec=10.0
lambda_mask=10.0
nepochs_no_decay=5  # fixing learning rate when epoch ranges in [0, 5]
nepochs_decay=15    # decreasing the learning rate when epoch ranges in [5, 15+5]

python3 train_densepose.py --gpu_ids ${gpu_ids}        \
    --holo_data_dir    ${holo_data_dir}     \
    --holo_images_folder    ${holo_images_folder}     \
    --holo_smpls_folder     ${holo_smpls_folder}      \
    --holo_train_ids_file   ${holo_train_ids_file}    \
    --holo_test_ids_file    ${holo_test_ids_file}     \
    --holo_uvs_folder       ${holo_uvs_folder}         \
    --checkpoints_dir  ${checkpoints_dir}   \
    --load_path        ${load_G_path}       \
    --n_threads_train  ${n_threads_train}   \
    --model            ${model}             \
    --gen_name         ${gen_name}          \
    --name             ${name}              \
    --dataset_mode     ${dataset_mode}     \
    --image_size       ${image_size}        \
    --batch_size       ${batch_size}        \
    --lambda_rec       ${lambda_rec}         \
    --lambda_mask      ${lambda_mask}       \
    --nepochs_no_decay ${nepochs_no_decay}  --nepochs_decay ${nepochs_decay}  \
    --uv_mapping ${uv_mapping} \
    --hmr_model ${hmr_model} \
    --smpl_model ${smpl_model} \
    --smpl_faces ${smpl_faces} \
    --face_model ${face_model} \
    --part_info ${part_info} \
    --front_info ${front_info} \
    --head_info ${head_info}
