import time
from lwganrt.options.train_options import TrainOptions
from lwganrt.data.custom_dataset_data_loader import CustomDatasetDataLoader
from lwganrt.models.models import ModelsFactory
from lwganrt.utils.tb_visualizer import TBVisualizer
from collections import OrderedDict


class Train(object):
    def __init__(self,args):
        self._opt = args
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        print('#train video clips = %d' % self._dataset_train_size)
        print('#test video clips = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        # self._train()
        self._save_internals()

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)

            # save model
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
            self._model.save(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals

            # train model
            self._model.set_input(train_batch)
            trainable = ((i_train_batch+1) % self._opt.train_G_every_n_iterations == 0) or do_visuals
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals, trainable=trainable)

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            # display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, do_visuals)
                self._last_print_time = time.time()

            # display visualizer
            if do_visuals:
                self._display_visualizer_train(self._total_steps)
                self._display_visualizer_val(i_epoch, self._total_steps)
                self._last_display_time = time.time()

            # save model
            if self._last_save_latest_time is None or time.time() - self._last_save_latest_time > self._opt.save_latest_freq_s:
                print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
                self._model.save(i_epoch)
                self._last_save_latest_time = time.time()

            if i_train_batch % 50 == 0:
                self._model.save_textures(i_epoch, i_train_batch)


    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors, t, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._dataset_test):
            if i_val_batch == self._opt.num_iters_validate:
                break

            # evaluate model
            self._model.set_input(val_batch)
            self._model.forward(keep_data_for_visuals=(i_val_batch == 0))
            errors = self._model.get_current_errors()

            # store current batch errors
            for k, v in errors.items():
                if k in val_errors:
                    val_errors[k] += v
                else:
                    val_errors[k] = v

        # normalize errors
        for k in val_errors:
            val_errors[k] /= self._opt.num_iters_validate

        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=False)

        # set model back to train
        self._model.set_train()

    def _save_internals(self):
        import os
        import numpy as np
        import cv2
        root_dir = '/home/darkalert/builds/ImpersonatorRT/lwganrt/outputs/test'
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            self._model.set_input(train_batch)
            fake_src_imgs, fake_tsf_imgs, fake_masks, debug_data = self._model.forward()

            fake_src_imgs = fake_src_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            fake_src_imgs = (fake_src_imgs+1.0) / 2.0 *255.0
            fake_src_imgs = fake_src_imgs.astype(np.uint8)#[...,::-1]

            fake_tsf_imgs = fake_tsf_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            fake_tsf_imgs = (fake_tsf_imgs+1.0) / 2.0 *255.0
            fake_tsf_imgs = fake_tsf_imgs.astype(np.uint8)#[...,::-1]

            fake_masks = fake_masks.permute(0, 2, 3, 1).detach().cpu().numpy()
            v_max = np.max(fake_masks)
            v_min = np.min(fake_masks)
            print (v_max, v_min)
            fake_masks = fake_masks*255.0
            fake_masks = fake_masks.astype(np.uint8)#[...,::-1]

            src_patches = debug_data['src_patches'].permute(0, 2, 3, 1).detach().cpu().numpy()
            v_max = np.max(src_patches)
            v_min = np.min(src_patches)
            print('src_patches',v_max, v_min)
            src_patches = (src_patches+1.0) / 2.0 *255.0
            src_patches = src_patches.astype(np.uint8)#[...,::-1]

            tsf_patches = debug_data['tsf_patches'].permute(0, 2, 3, 1).detach().cpu().numpy()
            v_max = np.max(tsf_patches)
            v_min = np.min(tsf_patches)
            print('tsf_patches',v_max, v_min)
            tsf_patches = (tsf_patches+1.0) / 2.0 *255.0
            tsf_patches = tsf_patches.astype(np.uint8)#[...,::-1]

            src_uv = debug_data['src_uv'].detach().cpu().numpy()
            v_max = np.max(src_uv)
            v_min = np.min(src_uv)
            print('src_uv',v_max, v_min)
            src_uv = (src_uv+1.0) / 2.0 *255.0
            src_uv = src_uv.astype(np.uint8)#[...,::-1]

            tsf_uv = debug_data['tsf_uv'].detach().cpu().numpy()
            v_max = np.max(tsf_uv)
            v_min = np.min(tsf_uv)
            print('tsf_uv',v_max, v_min)
            tsf_uv = (tsf_uv+1.0) / 2.0 *255.0
            tsf_uv = tsf_uv.astype(np.uint8)#[...,::-1]

            src_grid = debug_data['src_grid'].permute(0, 1, 3, 4, 2).detach().cpu().numpy()
            v_max = np.max(src_grid)
            v_min = np.min(src_grid)
            print('src_grid', v_max, v_min)
            src_grid = (src_grid + 1.0) / 2.0 * 255.0
            src_grid = src_grid.astype(np.uint8)


            bs = fake_src_imgs.shape[0]
            for i in range(bs):
                for j in range(24):
                    dst_path = os.path.join(root_dir, 'src_grid_' + str(i_train_batch) + '_' + str(i) + '_part' + str(j)  + '.png')
                    img = cv2.cvtColor(src_grid[i,j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst_path, img)


                # dst_path = os.path.join(root_dir, 'fake_src_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = cv2.cvtColor(fake_src_imgs[i], cv2.COLOR_RGB2BGR)
                # cv2.imwrite(dst_path, img)
                #
                # dst_path = os.path.join(root_dir, 'fake_tsg_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = cv2.cvtColor(fake_tsf_imgs[i], cv2.COLOR_RGB2BGR)
                # cv2.imwrite(dst_path, img)
                #
                # dst_path = os.path.join(root_dir, 'facke_mask_src_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = fake_masks[i,:,:,0]
                # cv2.imwrite(dst_path, img)
                #
                # dst_path = os.path.join(root_dir, 'facke_mask_tsf_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = fake_masks[i,:,:,1]
                # cv2.imwrite(dst_path, img)
                #
                # dst_path = os.path.join(root_dir, 'src_patches_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = cv2.cvtColor(src_patches[i], cv2.COLOR_RGB2BGR)
                # cv2.imwrite(dst_path, img)
                #
                # dst_path = os.path.join(root_dir, 'tsf_patches_' + str(i_train_batch) + '_' + str(i) + '.png')
                # img = cv2.cvtColor(tsf_patches[i], cv2.COLOR_RGB2BGR)
                # cv2.imwrite(dst_path, img)
                #
                for j in range(24):
                    dst_path = os.path.join(root_dir, 'src_uv_U_' + str(i_train_batch) + '_' + str(i) + '_part' + str(j)  + '.png')
                    img = src_uv[i,j,:,:,0]
                    cv2.imwrite(dst_path, img)

                    dst_path = os.path.join(root_dir,'src_uv_V_' + str(i_train_batch) + '_' + str(i) + '_part' + str(j) + '.png')
                    img = src_uv[i, j, :, :, 1]
                    cv2.imwrite(dst_path, img)
                #
                #     dst_path = os.path.join(root_dir,'tsf_uv_U_' + str(i_train_batch) + '_' + str(i) + '_part' + str(j) + '.png')
                #     img = tsf_uv[i, j, :, :, 0]
                #     cv2.imwrite(dst_path, img)
                #
                #     dst_path = os.path.join(root_dir,
                #                             'tsf_uv_V_' + str(i_train_batch) + '_' + str(i) + '_part' + str(j) + '.png')
                #     img = tsf_uv[i, j, :, :, 1]
                #     cv2.imwrite(dst_path, img)


            break


if __name__ == "__main__":
    args = TrainOptions().parse()

    # args.gpu_ids = '0'               # if using multi-gpus, increasing the batch_size
    #
    # # dataset configs
    # args.dataset_mode = 'Holo'
    # args.data_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data'      # need to be replaced!!!!!
    # args.images_folder = 'avatars'
    # args.smpls_folder = 'smpl_aligned_lwgan'
    # args.train_ids_file = 'train.txt'
    # args.test_ids_file = 'val.txt'
    #
    # # saving configs
    # args.checkpoints_dir = '/home/darkalert/builds/impersonator/outputs'   # directory to save models, need to be replaced!!!!!
    # args.name = 'Holo1'               # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.
    #
    # # model configs
    # args.model = 'holoportator_trainer'
    # args.gen_name = 'holoportator'
    # args.image_size = 256
    #
    # # training configs
    # args.load_path = 'None'
    # args.batch_size = 8
    # args.lambda_rec = 10.0
    # args.lambda_tsf = 10.0
    # args.lambda_face = 5.0
    # args.lambda_style = 0.0
    # args.lambda_mask = 1.0
    # args.lambda_mask_smooth = 1.0
    # args.nepochs_no_decay = 5         # fixing learning rate when epoch ranges in [0, 5]
    # args.nepochs_decay = 25           # decreasing the learning rate when epoch ranges in [6, 25+5]
    #
    # args.mask_bce = True
    # args.use_vgg = True
    # args.use_face = True

    Train(args)

