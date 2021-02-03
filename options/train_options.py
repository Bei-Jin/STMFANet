import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='KTH', help='name of experiment')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        self.parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size")
        self.parser.add_argument('--image_size', type=int, default=128, help='image size')


        self.parser.add_argument('--K', type=int, default=10, help='Number of frames to observe from the past')
        self.parser.add_argument('--T', type=int, default=10, help='Number of frames to predict')
        self.parser.add_argument('--c_dim', type=int, default=1, help='# of image channels')

        self.parser.add_argument('--model', type=str, default='STMF', help='name of model')
        self.parser.add_argument('--depth', type=int, default=22, help='layers of one RRDB')
        self.parser.add_argument('--growthRate', type=int, default=16, help='# of filters to add per dense block')
        self.parser.add_argument('--reduction', type=float, default=0.5,
                                 help='reduction factor of transition blocks. Note : reduction value is inverted to compute compression')
        self.parser.add_argument('--bottleneck', type=bool, default=True, help='use bottleneck or not')
        self.parser.add_argument('--gf_dim', type=int, default=16, help='base number of channels')


        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved in this folder')
        self.parser.add_argument('--tensorboard_dir', type=str, default='./tb', help='for tensorboard visualization')
        self.parser.add_argument('--txtroot', type=str, default='./data/', help='location of data txt file, need to set')
        self.parser.add_argument('--data_root', type=str, default='./data', help='data path')

        self.parser.add_argument("--lr", type=float, default=0.0001, help="Base Learning Rate")
        self.parser.add_argument('--nepoch', type=int, default=100, help='# of epoch at starting learning rate')
        self.parser.add_argument('--nepoch_decay', type=str, default=100, help='# of epoches at starting learning rate')
        self.parser.add_argument('--continue_train', type=bool, default=False, help='continue train')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='laod which epoch')
        self.parser.add_argument("--alpha", type=float, dest="alpha", default=1.0, help="Image loss weight")
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--adversarial', action='store_true', help='do not use the adversarial loss')
        self.parser.add_argument('--lr_policy', type=str, default='step',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--start_epoch', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--print_freq', type=int, default=1000,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--display_freq', type=int, default=1000,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,
                                 help='frequency of saving the latest results')

        self.is_train = True

        self.initialized = True

    def parse(self):

        self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if self.opt.is_train:
            self.opt.video_list = 'train_data_list.txt'
            return self.opt
        else:
            return self.opt
