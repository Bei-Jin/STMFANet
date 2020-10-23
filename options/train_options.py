from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--batch_size", type=int, dest="batch_size", default=8, help="Mini-batch size")
        self.parser.add_argument("--lr", type=float, dest="lr", default=0.0001, help="Base Learning Rate")
        self.parser.add_argument("--alpha", type=float, dest="alpha", default=1.0, help="Image loss weight")
        self.parser.add_argument("--beta", type=float, dest="beta", default=0.01, help="GAN loss weight")
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', default='True', help='continue training')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam')
        self.parser.add_argument('--lr_policy', type=str, default= 'lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--margin', type=float, default=0.3, help="the margin used for choosing opt D or G")
        self.parser.add_argument('--nepoch', type=int, default=100, help='# of epoch at starting learning rate')
        self.parser.add_argument('--nepoch_decay', type=int, default=100, help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--D_G_switch', type=str, default='adaptive', help='type of switching training in D and G [adaptive|alternative]')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--no_adversarial', action='store_true', help='do not use the adversarial loss')
        
        # Data Augment
        self.parser.add_argument('--data', required=True, type=str, help="name of training dataset")
        self.parser.add_argument("--backwards", default=True, type=bool, help="play the video backwards")
        self.parser.add_argument("--pick_mode", default='Random', type=str, help="pick up clip [Random|First|Sequential]")
        self.parser.add_argument("--flip", default=True, type=bool, help="flip the frames in the videos")

        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.is_train = True
