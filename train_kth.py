from options.train_options import TrainOptions
import os
from joblib import Parallel, delayed
from wavenet_models.create_model import create_model
from util import util
from util import visualizer
import numpy as np
import torch
from tensorboardX import SummaryWriter
import time
def main():
    opt = TrainOptions().parse()
    #read training files
    f = open(os.path.join(opt.txtroot, opt.video_list), 'r')
    trainfiles = f.readlines()
    print('video num: %s' %len(trainfiles))


    #create model
    model = create_model(opt)
    total_steps = 0
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    with Parallel(n_jobs = opt.batch_size) as parallel:
        for epoch in range(opt.start_epoch, opt.nepoch + opt.nepoch_decay + 1):
            mini_batches = util.get_minibatches_idx(len(trainfiles), opt.batch_size, shuffle=True)

            for _, batchidx  in mini_batches:
                if len(batchidx) == opt.batch_size:
                    inputs_batch = np.zeros((opt.batch_size, 1, opt.image_size, opt.image_size, opt.K + opt.T), dtype='float32')

                    Ts = np.repeat(np.array([opt.T]), opt.batch_size, axis = 0)
                    Ks = np.repeat(np.array([opt.K]), opt.batch_size, axis=0)
                    paths = np.repeat(opt.data_root, opt.batch_size, axis=0)
                    tfiles = np.array(trainfiles)[batchidx]
                    shapes = np.repeat(np.array([opt.image_size]), opt.batch_size, axis=0)
                    output = parallel(delayed(util.load_kth_data)(f, p, image_size, k, t) for f, p, image_size, k, t in
                                      zip(tfiles, paths, shapes, Ks, Ts))
                    output = torch.stack(output, dim=0)
                    model.set_inputs(output)
                    model.optimize_parameters()
                    total_steps += 1

                    if total_steps % opt.print_freq == 0:
                        print('total_steps % opt.print_freq == 0')
                        errors = model.get_current_errors()

                        for key in errors.keys():
                            writer.add_scalar('loss/%s' % (key), errors[key], total_steps / opt.batch_size)

                        util.print_current_errors(epoch, total_steps, errors, opt.checkpoints_dir, opt.name)
                    if total_steps % opt.display_freq == 0:
                        print('total_steps % opt.display_freq == 0')
                        visuals = model.get_current_visuals()
                        grid = util.visual_grid(visuals['seq_batch'], visuals['pred'], opt.K, opt.T)
                        writer.add_image('current_batch', grid, total_steps / opt.batch_size)
                    if total_steps % opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_steps %d)' %
                              (epoch, total_steps))
                        model.save('latest', epoch)

        print("end training")



if __name__ == "__main__":
    main()
