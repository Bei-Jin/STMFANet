#This part of the code refers to the work of https://github.com/sunxm2357/mcnet_pytorch/. Thanks!
import argparse
import os
from util.util import *
import torch
import copy

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment.')
        self.parser.add_argument("--K", type=int, dest="K", default=10, help="the length of observation")
        self.parser.add_argument("--T", type=int, dest="T", default=10, help="the length of prediction")
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        # self.parser.add_argument("--prefix", type=str, dest="prefix", required=True, help="Prefix for log/snapshot")
        self.parser.add_argument('--c_dim', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--gf_dim', type=int, default=64, help='# of gen filters')
        self.parser.add_argument('--df_dim', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--growthRate', type=int, default=12, help='# of filters to add per dense block')
        self.parser.add_argument('--depth', type=int, default=22, help='# of layers')
        self.parser.add_argument('--reduction', type=float, default=0.5, help='reduction factor of transition blocks. Note : reduction value is inverted to compute compression')
        self.parser.add_argument('--bottleneck', type=bool, default=True, help='whether use bottleneck layer')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--tensorboard_dir', type=str, default='./tb', help='models are saved here')
        self.parser.add_argument('--visualize_dir', type=str, default='./temp', help='temporary results are saved here')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        self.parser.add_argument('--init_type', type=str, default='wavenet', help='network initialization')
        self.parser.add_argument("--debug", default=False, type=bool, help="when debugging, overfit to the first training samples")
        self.parser.add_argument('--model', required=True, help='the model to run')

        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument("--image_size", required=True, help="image size")
     
        self.parser.add_argument('--dataroot', required=True, help='path to videos')
        self.parser.add_argument('--textroot', required=True, help='path to trainings')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.initialized = True
        def parse(self):
            if not self.initialized:
                self.initialize()
            self.opt = self.parser.parse_args()
            self.opt.is_train = self.is_train   # train or test
            if len(self.opt.image_size) == 1:
                a = self.opt.image_size[0]
                self.opt.image_size.append(a)
            
            str_ids = self.opt.gpu_ids.split(',')
            self.opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    self.opt.gpu_ids.append(id)
            # set gpu ids
            if len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])

            args = vars(self.opt)
            
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
            
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            makedir(expr_dir)
            vis_dir = os.path.join(self.opt.visualize_dir, self.opt.name)
            makedir(vis_dir)
            tb_dir = os.path.join(self.opt.tensorboard_dir, self.opt.name)
            makedir(tb_dir)
            
            if not self.is_train:
                self.opt.serial_batches = True
                self.opt.video_list = 'test_data_list.txt'
                self.opt.quant_dir = os.path.join(self.opt.result_dir, 'quantitative', self.opt.data, self.opt.name + '_' + self.opt.which_epoch)
                makedir(self.opt.quant_dir)
                self.opt.save_dir = os.path.join(self.opt.result_dir, 'images', self.opt.data, self.opt.name + '_' + self.opt.which_epoch)
                makedir(self.opt.save_dir)
                
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
                            
            if self.is_train:
                self.opt.video_list = 'train_data_list.txt'
                if self.opt.debug:
                    self.opt.print_freq = 1
                    self.opt.save_epoch_freq = 1
                    self.opt.display_freq = 1
                    self.opt.pick_mode = "First"
                    self.opt.save_latest_freq = 1
                self.val_opt = copy.deepcopy(self.opt)
                self.val_opt.serial_batches = True
                self.val_opt.video_list = 'val_data_list.txt'
                self.val_opt.batch_size = 1
                self.val_opt.is_train = False
                self.val_opt.which_epoch = 'latest'
                if self.opt.data == 'KTH':
                    self.val_opt.pick_mode = 'Slide'
                    self.val_opt.T = self.opt.T * 2
                
                if self.opt.debug:
                    self.val_opt.pick_mode = 'First'
                return self.opt, self.val_opt
            else:
                return self.opt  

              
