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
        
        
