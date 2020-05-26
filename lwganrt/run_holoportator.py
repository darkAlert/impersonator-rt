import os
import numpy as np
import torch
from lwganrt.models.holoportator import Holoportator
from lwganrt.options.test_options import TestOptions


def get_file_paths(path, exts=('.jpeg','.jpg','.png')):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if any(f.endswith(ext) for ext in exts)]:
            file_paths.append(os.path.join(dirpath,filename))
            file_paths.sort()

    return file_paths


def prepare_test_data(frames_dir, smpls_dir):
    # Parse frame paths:
    subdirs = [p.path for p in os.scandir(frames_dir) if p.is_dir()]
    subdirs.sort()
    frame_paths = {}

    for s_dir in subdirs:
        cam_name = s_dir.split('/')[-1]
        paths = get_file_paths(s_dir)
        frame_paths[cam_name] = paths

    # Load smpls:
    subdirs = [p.path for p in os.scandir(smpls_dir) if p.is_dir()]
    subdirs.sort()
    smpls = {}

    for s_dir in subdirs:
        smpl_path = get_file_paths(s_dir, exts=('.npz'))[0]
        with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
            smpl_data = dict(data)
            n = len(smpl_data['cams'])
            smpl_vecs = []

            for frame_id in range(n):
                cams = smpl_data['cams'][frame_id]
                pose = smpl_data['pose'][frame_id]
                shape = smpl_data['shape'][frame_id]
                vec = np.concatenate((cams, pose, shape), axis=0)
                vec = np.expand_dims(vec, axis=0)
                smpl_vecs.append(torch.tensor(vec).float().cuda())

            cam_name = s_dir.split('/')[-1]
            smpls[cam_name] = torch.cat(smpl_vecs, dim=0)

    # Check:
    assert len(smpls) == len(frame_paths)
    n = len(list(frame_paths.values())[0])
    assert (n == len(v) for v in frame_paths.values())
    m = list(smpls.values())[0].shape[0]
    assert (m == smpl.shape[0] for smpl in smpls.values())
    assert n == m
    print ('Data has been loaded from path:', frames_dir)
    print ('Total cams={}, frames={}'.format(len(smpls), n))

    test_data = {
        'frame_paths' : frame_paths,
        'smpls' : smpls
    }

    return test_data, n


def holoportate(test_opt, one_shot=False):
    # Load and prepare test data:
    smpls_dir = os.path.join(test_opt.root_dir, test_opt.smpl_dir, test_opt.scene_path)
    frames_dir = os.path.join(test_opt.root_dir, test_opt.frames_dir, test_opt.scene_path)
    test_data, n = prepare_test_data(frames_dir, smpls_dir)

    if test_opt.frames_range[1] < test_opt.frames_range[0]:
        test_opt.frames_range = (test_opt.frames_range[0],n)

    # Determine target camera names:
    tgt_cams = []
    for cam_name in test_data['frame_paths'].keys():
        if cam_name != test_opt.src_cam:
            tgt_cams.append(cam_name)

    # Prepare output dir:
    output_dir = os.path.join(test_opt.root_dir, test_opt.output_dir)
    for cam_name in tgt_cams + [test_opt.src_cam]:
        path = os.path.join(output_dir, cam_name)
        if not os.path.exists(path):
            os.makedirs(path)

    # Init Holoportator:
    holoport = Holoportator(test_opt)
    personalized = False

    for idx in range(test_opt.frames_range[0], test_opt.frames_range[1]):
        print ('Processing frame', idx)

        # Personalize model:
        if not personalized:
            img_path = test_data['frame_paths'][test_opt.src_cam][idx]
            smpl = test_data['smpls'][test_opt.src_cam][idx].unsqueeze(0)
            out_path_src = os.path.join(output_dir, test_opt.src_cam, str(idx).zfill(5) + '.jpeg')
            holoport.personalize(src_path=img_path, src_smpl=smpl, output_path=out_path_src)

        if one_shot:
            personalized = True

        # Inference:
        tgt_smpls, out_paths = [], []
        for cam_name in tgt_cams:
            tgt_smpls.append(test_data['smpls'][cam_name][idx].unsqueeze(0))
            out_paths.append(os.path.join(output_dir, cam_name, str(idx).zfill(5) + '.jpeg'))
        holoport.inference_by_smpls(tgt_smpls=tgt_smpls, cam_strategy='', output_dir=out_paths)

    print ('All done!')


if __name__ == "__main__":
    # Parse options:
    test_opt = TestOptions().parse()

    # # Data (Holo):
    # test_opt.root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data'
    # test_opt.frames_dir = 'avatars'
    # test_opt.smpl_dir = 'smpls_by_vibe_aligned_lwgan'
    #
    # test_opt.scene_path = 'person_9/light-100_temp-5600/garments_2/rotation'
    # test_opt.output_dir = 'test/holoportator/holo2-i30_p9-l100-g2-rotation'
    # test_opt.src_cam = 'cam1'
    # test_opt.frames_range = (0,-1)

    # Data (iPER)::
    test_opt.root_dir = '/home/darkalert/KazendiJob/Data/iPER/Data'
    test_opt.frames_dir = 'avatars'
    test_opt.smpl_dir = 'smpls_by_vibe_lwgan'

    test_opt.scene_path = '010/1'
    test_opt.output_dir = 'test/holoport_view/view_010-1_h3e20_1shot'
    test_opt.src_cam = '1'
    test_opt.frames_range = (0,-1)

    # Model:
    test_opt.gpu = '0'
    test_opt.gen_name = "holoportator"
    test_opt.image_size = 256
    test_opt.checkpoints_dir = './outputs/checkpoints'
    test_opt.load_path = '/home/darkalert/builds/impersonator/outputs/Holo_iPER/net_epoch_20_id_G.pth'
    test_opt.bg_ks = 11
    test_opt.ft_ks = 3
    test_opt.has_detector = False
    test_opt.post_tune = False
    test_opt.front_warp = False
    test_opt.save_res = True

    holoportate(test_opt, one_shot=True)

