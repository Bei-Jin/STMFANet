import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.batch_size = opt.batch_size
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.video_dir = os.path.join(opt.visualize_dir, opt.name)
        self.loss_plot = os.path.join(opt.checkpoints_dir, opt.name, 'loss.png')
        with open(self.log_name, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('=================== Train Loss (%s)==================\n' % now)

    def plot_current_errors(self, epoch, counter_ratio, errors):
        plt.clf()
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}

        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        plt.plot(self.plot_data['X'], self.plot_data['Y'])
        plt.legend(self.plot_data['legend'])
        plt.title(self.name+ ' loss over time')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.loss_plot)

    def print_current_errors(self, epoch, i, errors):
        message = 'epoch: %d, iters: %d' % (epoch, i)
        for k, v in errors.items():
            if k.startswith('Update'):
                message += '%s: %s ' % (k, str(v))
            else:
                message += '%s: %.3f ' %(k,v)

        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s \n' % message)

    def save_images(self, visuals, video_name, epoch, iter):
        """
        :param visuals: dict, diff_in: K-1 [batch_size, h, w, c], ndarray, [0,255]; targets: K+T ndarrays, pred: T ndarray
        :param video_name: batch_size strs
        :param epoch: current epoch
        :param iter: current iter in this epoch
        :return: no return
        """
        video_folder = os.path.join(self.video_dir, 'epoch%s_iter%s' % (epoch, iter))
        os.makedir(video_folder)
        for i in range(self.batch_size):
            title = video_name[i].split('.')[0]
            single_video = os.path.join(video_folder, title)
            os.makedir(single_video)
            for key in visuals.keys():
                key_folder = os.path.join(single_video, key)
                os.makedir(key_folder)
                l = len(visuals[key])
                for t in range(l):
                    out_name = os.path.join(key_folder, str(t).zfill(3)+'.png')
                    cv2.imwrite(out_name, visuals[key][t][i])

