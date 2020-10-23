#This part of the code refers to the work of https://github.com/sunxm2357/mcnet_pytorch/. Thanks!
import time
from options.train_options import TrainOptions
from wavenet_models.models import create_model
from data.data_loader import *
from util.visualizer import Visualizer
import pdb
from tensorboardX import SummaryWriter
from val import *
import traceback
from joblib import Parallel, delayed


def main():
    opt, val_opt = TrainOptions().parse()
    data_path = opt.dataroot
    f = open(os.path.join(opt.textroot, opt.video_list), 'r')
    trainfiles = f.readlines()
    one_epoch = len(trainfiles) # num of iters in one epoch
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    with Parallel(n_jobs=opt.batch_size) as parallel:
        epoch_start_time = time.time()
        epoch_iters = 0 
        
        for epoch in range(model.start_epoch, opt.nepoch + opt.nepoch_decay + 1):
            mini_batches = get_minibatches_idx(len(trainfiles), opt.batch_size, shuffle=True)
            for _, batchidx in mini_batches:
                if len(batchidx) == opt.batch_size:
                    total_steps += opt.batch_size
                    epoch_iters += opt.batch_size
                    iter_start_time = time.time()
                    Ts = np.repeat(np.array([opt.T]), opt.batch_size, axis=0)
                    Ks = np.repeat(np.array([opt.K]), opt.batch_size, axis=0)
                    paths = np.repeat(data_path, opt.batch_size, axis=0)
                    tfiles = np.array(trainfiles)[batchidx]
                    shapes = np.repeat(np.array([opt.image_size]), opt.batch_size, axis=0)
                    input_data = parallel(delayed(load_kth_data)(f, p, img_sze, k, t)
                                      for f, p, img_sze, k, t in zip(tfiles,
                                                                     paths,
                                                                     shapes,
                                                                     Ks, Ts))
                    input_data = torch.stack(input_data, dim=0)
                    model.set_inputs(input_data)
                    model.optimize_parameters()
                    
                    if total_steps % opt.print_freq == 0:
                        errors = model.get_current_errors()
                        t = (time.time() - iter_start_time) / opt.batch_size
                        writer.add_scalar('iter_time', t, total_steps / opt.batch_size)
                        
                        for key in errors.keys():
                            writer.add_scalar('loss/%s' % (key), errors[key], total_steps / opt.batch_size
                                              
                        visualizer.print_current_errors(epoch, epoch_iters, errors, t)
                                              
                    if total_steps % opt.display_freq == 0:
                        print('total_steps % opt.display_freq == 0')
                        visuals = model.get_current_visuals()
                        grid = visual_grid(visuals['seq_batch'], visuals['pred'], opt.K, opt.T)
                        writer.add_image('current_batch', grid, total_steps / opt.batch_size)
                                              
                    if total_steps % opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_steps %d)' %
                              (epoch, total_steps))
                        model.save('latest', epoch)

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save('latest', epoch)
                model.save(epoch, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

if __name__ == "__main__":
    main()




