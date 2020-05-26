import os
import numpy as np
import torch
from lwganrt.models.holoportator import Holoportator
from lwganrt.options.test_options import TestOptions
from tqdm import tqdm
import time


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


def parse_view_params(view_params):
    """
    :param view_params: R=xxx,xxx,xxx/t=xxx,xxx,xxx
    :return:
        -R: np.ndarray, (3,)
        -t: np.ndarray, (3,)
    """
    params = dict()

    for segment in view_params.split('/'):
        # R=xxx,xxx,xxx -> (name, xxx,xxx,xxx)
        name, params_str = segment.split('=')
        vals = [float(val) for val in params_str.split(',')]
        params[name] = np.array(vals, dtype=np.float32)
    params['R'] = params['R'] / 180 * np.pi

    return params


def holoport_view(test_opt, steps):
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
    output_dir = os.path.join(test_opt.root_dir, test_opt.output_dir, test_opt.src_cam)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse view params:
    params = parse_view_params(test_opt.view_params)
    delta = 360 / steps

    # Init Holoportator:
    holoport = Holoportator(test_opt)

    for idx in range(test_opt.frames_range[0], test_opt.frames_range[1]):
        print ('Processing frame', idx)

        # Personalize model:
        img_path = test_data['frame_paths'][test_opt.src_cam][idx]
        smpl = test_data['smpls'][test_opt.src_cam][idx].unsqueeze(0)
        name = str(idx).zfill(5) + '-00000.jpeg'
        out_path_src = os.path.join(output_dir, name)
        holoport.personalize(src_path=img_path, src_smpl=smpl, output_path=out_path_src)

        # Inference:
        logger = tqdm(range(steps))
        print('Synthesizing {} novel views...'.format(steps))
        for i in logger:
            # params['R'][0] = 10 / 180 * np.pi
            # params['R'][1] = delta * i / 180.0 * np.pi
            # params['R'][2] = 10 / 180 * np.pi
            params['R'][0] = 0
            params['R'][1] = delta * i / 180.0 * np.pi
            params['R'][2] = 0

            name =  str(idx).zfill(5) + '-' + str(i+1).zfill(5) + '.jpeg'
            out_path = os.path.join(output_dir, name)
            holoport.view(params['R'], params['t'], out_path)
            print ('view = ({:.3f}, {:.3f}, {:.3f})'.format(params['R'][0], params['R'][1], params['R'][2]))

    print ('All done!')


def holoport_view_seq(test_opt, steps, save_descriptor=False):
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
    output_dir = os.path.join(test_opt.root_dir, test_opt.output_dir, test_opt.src_cam)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse view params:
    params = parse_view_params(test_opt.view_params)
    delta = 360 / steps
    step_i = 0

    # Init Holoportator:
    holoport = Holoportator(test_opt)
    start = time.time()

    for idx in range(test_opt.frames_range[0], test_opt.frames_range[1]):
        print ('Processing frame', idx)

        # Personalize model:
        img_path = test_data['frame_paths'][test_opt.src_cam][idx]
        smpl = test_data['smpls'][test_opt.src_cam][idx].unsqueeze(0)
        holoport.personalize(src_path=img_path, src_smpl=smpl, output_path=None)
        if save_descriptor:
            holoport.save_descriptor(src_path=img_path, src_smpl=smpl, output_dir=output_dir, output_name=idx)

        # Inference:
        params['R'][0] = 0
        params['R'][1] = delta * step_i / 180.0 * np.pi
        params['R'][2] = 0

        name =  str(idx).zfill(5) + '.jpeg'
        out_path = os.path.join(output_dir, name)
        holoport.view(params['R'], params['t'], out_path)

        step_i += 1
        if step_i >= steps:
            step_i = 0

    end = time.time()
    print ('All done! Time per frame:', (end - start)/test_opt.frames_range[1]-test_opt.frames_range[0])


if __name__ == "__main__":
    # Parse options:
    test_opt = TestOptions().parse()

    # Data (Holo):
    test_opt.root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data'
    test_opt.frames_dir = 'avatars'
    test_opt.smpl_dir = 'smpls_by_vibe_aligned_lwgan'

    # test_opt.scene_path = 'person_2/light-100_temp-5600/garments_2/front_position'
    test_opt.scene_path = 'person_9/light-100_temp-5600/garments_1/freestyle'
    test_opt.output_dir = 'test/holoport_view/view_p2-l100-g2-front_h3e20'
    test_opt.src_cam = 'cam1'
    test_opt.frames_range = (0,-1)

    # # Data (iPER):
    # test_opt.root_dir = '/home/darkalert/KazendiJob/Data/iPER/Data'
    # # test_opt.frames_dir = 'iPER_1024_images'
    # test_opt.frames_dir = 'avatars'
    # test_opt.smpl_dir = 'smpls_by_vibe_lwgan'
    #
    # test_opt.scene_path = '001/1'
    # test_opt.output_dir = 'test/holoport_view/view_001-1_imper'
    # test_opt.src_cam = '2'
    # test_opt.frames_range = (0,-1)


    # Model:
    test_opt.gpu = '0'
    test_opt.gen_name = "holoportator"
    test_opt.image_size = 256
    test_opt.checkpoints_dir = './outputs/checkpoints'
    test_opt.load_path = '/home/darkalert/builds/impersonator/outputs/Holo_iPER/net_epoch_20_id_G.pth'
    # test_opt.load_path = '/home/darkalert/builds/impersonator/outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth'

    test_opt.bg_ks = 11
    test_opt.ft_ks = 3
    test_opt.has_detector = False
    test_opt.post_tune = False
    test_opt.front_warp = False
    test_opt.save_res = True
    # test_opt.has_detector = True

    # Novel view params:
    test_opt.view_params = 'R=0,0,0/t=0,0,0'

    # holoport_view(test_opt, steps=1)
    holoport_view_seq(test_opt, steps=1, save_descriptor=False)

