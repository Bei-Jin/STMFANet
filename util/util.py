import random
import numpy as np
import torchvision.utils as vutils

def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n):
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


import cv2
import imageio
import torchvision.transforms as transforms
import torch
def load_kth_data(f_name, data_path, image_size, K, T):

    inputs = []
    flip = np.random.binomial(1,.5,1)[0]
    tokens = f_name.split()
    vid_path = data_path + tokens[0] + "_uncomp.avi"
    vid = imageio.get_reader(vid_path, "ffmpeg")
    low = int(tokens[1])

    high = np.min([int(tokens[2]), vid.get_length()]) - K - T + 1
    if low == high:
        stidx = 0
    else:
        if low >= high: print(vid_path)
        stidx = np.random.randint(low=low, high=high)
    for t in range(K+T):
        c = 0
        while True:
            try:
                global img
                img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (128, 128)),
                                   cv2.COLOR_RGB2GRAY)
                break
            except Exception:
                c = c + 1
                if c > 5: break
                print("imageio failed loading frames, retrying")
        # if DEBUG:
        #     pdb.set_trace()
        assert (np.max(img) > 1, "the range of image should be [0,255]")
        if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
        if flip == 1:
            img = img[:, ::-1, :]
        # pdb.set_trace()
        inputs.append(transforms.ToTensor()(img.copy()))
    inputs = fore_transform(torch.stack(inputs, dim=-1))
    return inputs

def fore_transform(images):
    return images * 2 - 1


def inverse_transform(images):
    return (images+1.)/2


def visual_grid(seq_batch, pred, K, T):
    pred_data = torch.stack(pred, dim=-1)

    true_data = seq_batch[:, :, :, :, K:K + T].clone()

    pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data], dim=-1)
    true_data = torch.cat((seq_batch[:, :, :, :, :K], true_data), dim=-1)
    batch_size = int(pred_data.size()[0])
    c_dim = int(pred_data.size()[1])
    vis = []
    for i in range(batch_size):
        # pdb.set_trace()
        pred_data_sample = inverse_transform(pred_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        target_sample = inverse_transform(true_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        if c_dim == 1:
            pred_data_sample = torch.cat([pred_data_sample]*3, dim=1)
            target_sample = torch.cat([target_sample]*3, dim=1)
        pred_data_sample = draw_frame_tensor(pred_data_sample, K, T)
        target_sample = draw_frame_tensor(target_sample, K, T)
        output = torch.cat([pred_data_sample, target_sample], dim=0)
        vis.append(vutils.make_grid(output, nrow=K+T))
    grid = torch.cat(vis, dim=1)
    # pdb.set_trace()
    grid = torch.from_numpy(np.flip(grid.numpy(), 0).copy())
    return grid

def draw_frame_tensor(img, K, T):
    img[:K, 0, :2, :] = img[:K, 2, :2, :] = 0
    img[:K, 0, :, :2] = img[:K, 2, :, :2] = 0
    img[:K, 0, -2:, :] = img[:K, 2, -2:, :] = 0
    img[:K, 0, :, -2:] = img[:K, 2, :, -2:] = 0
    img[:K, 1, :2, :] = 1
    img[:K, 1, :, :2] = 1
    img[:K, 1, -2:, :] = 1
    img[:K, 1, :, -2:] = 1
    img[K:K+T, 0, :2, :] = img[K:K+T, 1, :2, :] = 0
    img[K:K+T, 0, :, :2] = img[K:K+T, 1, :, :2] = 0
    img[K:K+T, 0, -2:, :] = img[K:K+T, 1, -2:, :] = 0
    img[K:K+T, 0, :, -2:] = img[K:K+T, 1, :, -2:] = 0
    img[K:K+T, 2, :2, :] = 1
    img[K:K+T, 2, :, :2] = 1
    img[K:K+T, 2, -2:, :] = 1
    img[K:K+T, 2, :, -2:] = 1
    return img

import os
def print_current_errors(epoch, i, errors, checkpoints_dir, name):
    log_name = os.path.join(checkpoints_dir, name, 'loss_log.txt')
    message = 'epoch: %d, iters: %d' % (epoch, i)
    for k, v in errors.items():
        if k.startswith('Update'):
            message += '%s: %s ' % (k, str(v))
        else:
            message += '%s: %.3f ' %(k,v)

    print(message)
    with open(self.log_name, 'a') as log_file:
        log_file.write('%s \n' % message)
