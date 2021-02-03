
import torch
import torch.nn as nn
import math
from pytorch_wavelets import DWTForward

def define_generator(opt):
    use_gpu = len(opt.gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())

    generator = Generator(opt)
    if use_gpu:
        generator.cuda(device=opt.gpu_ids[0])
    init_weights(generator, init_type='STMF')
    return generator



class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.c_dim = opt.c_dim
        self.K = opt.K
        self.T = opt.T
        self.batch_size = opt.batch_size
        self.image_size = [opt.image_size, opt.image_size]

        self.encoder = define_encoder(opt)
        #num_features = 128
        self.convLstm_cell = define_convLstm_cell(feature_size=3, num_features = opt.gf_dim * 8, gpu_ids=self.gpu_ids)
        self.decoder = define_decoder(c_dim=opt.c_dim, gf_dim=opt.gf_dim, gpu_ids=self.gpu_ids)

    def forward(self, inputs, state):
        for k in range(self.K):
            h_encoder = self.encoder.forward(inputs[k])
            h_dyn, state = self.convLstm_cell.forward(h_encoder, state)

        pred=[]
        for t in range(self.T):
            if t>0:
                h_encoder = self.encoder.forward(xt)
                h_dyn, state = self.convLstm_cell.forward(h_encoder, state)

            x_hat = self.decoder.forward(h_dyn)

            xt = x_hat
            pred.append(x_hat.view(self.batch_size, self.c_dim, self.image_size[0], self.image_size[1]))
        return pred


def define_encoder(opt, init_type = 'STMF'):
    use_gpu = len(opt.gpu_ids)>0
    if use_gpu:
        assert (torch.cuda.is_available())

    encoder = Encoder(opt)
    if use_gpu:
        encoder.cuda(device = opt.gpu_ids[0])
    init_weights(encoder, init_type=init_type)
    return encoder

def define_convLstm_cell(feature_size, num_features, forget_bias=1, gpu_ids=[], init_type="STMF",
                         bias=True):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    convLstm_cell = ConvLstmCell(feature_size, num_features, gpu_ids, forget_bias=forget_bias,
                                 bias=bias)

    if len(gpu_ids) > 0:
        convLstm_cell.cuda(device=gpu_ids[0])

    init_weights(convLstm_cell, init_type=init_type)
    return convLstm_cell

def define_decoder(c_dim, gf_dim, init_type="STMF", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    decoder = Decoder(c_dim, gf_dim, gpu_ids)

    if len(gpu_ids) > 0:
        decoder.cuda(device=gpu_ids[0])
    init_weights(decoder, init_type=init_type)
    return decoder



class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        nDenseBlocks = (opt.depth-4) // 3  #(22-4)//3=6

        if opt.bottleneck:
            nDenseBlocks = nDenseBlocks // 2

        #define input and output channel
        nOut_conv1 = opt.growthRate

        nIn_dense1 = nOut_conv1
        nOut_dense1 = nOut_conv1 + nDenseBlocks * opt.growthRate

        nIn_trans1 = nOut_dense1
        nOut_trans1 = int(math.floor(nOut_dense1 * opt.reduction))

        nIn_dense2 = nOut_trans1
        nOut_dense2 = nOut_trans1 + nDenseBlocks * opt.growthRate

        nIn_trans2 = nOut_dense2
        nOut_trans2 = int(math.floor(nOut_dense2 * opt.reduction))

        nIn_dense3 = nOut_trans2
        nOut_dense3 = nOut_trans2 + nDenseBlocks * opt.growthRate

        nIn_trans3 = nOut_dense3
        nOut_trans3 = int(math.floor(nOut_dense3 * opt.reduction))

        #model blocks
        self.dwt = Dwtconv(nOut_trans1, nOut_trans2, nOut_trans3)
        self.conv1 = nn.Conv2d(opt.c_dim, nOut_conv1, kernel_size=(3,3), bias=True, padding=(1,1), stride=(1, 1),)
        self.dense1 = self._make_dense(nIn_dense1, opt.growthRate, nDenseBlocks, opt.bottleneck)
        self.trans1 = Transition(nIn_trans1, nOut_trans1)
        self.dense2 = self._make_dense(nIn_dense2, opt.growthRate, nDenseBlocks, opt.bottleneck)
        self.trans2 = Transition(nIn_trans2, nOut_trans2)
        self.dense3 = self._make_dense(nIn_dense3, opt.growthRate, nDenseBlocks, opt.bottleneck)
        self.trans3 = Transition(nIn_trans3, nOut_trans3)

        nChannels = nOut_trans3
        nOutChannels = opt.gf_dim * 8
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(nOutChannels)
        self.relu_out = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out_dwt0, out_dwt1, out_dwt2 = self.dwt(x)
        out = self.dense1(out)
        out = self.trans1(out)
        #print('out dwt1', out_dwt0.shape)
        out = self.dense2(torch.add(out_dwt0, out))

        out = self.trans2(out)
        out = self.dense3(torch.add(out_dwt1, out))
        out = self.trans3(out)
        out = self.conv2(self.relu1(self.bn1(torch.add(out_dwt2, out))))
        out = self.relu_out(self.bn_out(out))
        return out




    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 *growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.avg_pool(out)
        return out

class Dwtconv(nn.Module):
    def __init__(self, outC1, outC2, outC3):
        super(Dwtconv, self).__init__()

        self.dwt1 = DWTForward(J=1, wave='haar', mode='symmetric')
        outChannel_conv1 = outC1 // 2
        nIn_conv1 = 4
        self.conv1_1 = nn.Conv2d(nIn_conv1, outChannel_conv1, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(outChannel_conv1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(outChannel_conv1, outC1, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(outC1)
        self.relu1_2 = nn.ReLU()

        nIn_conv2 = 4
        self.dwt2 = DWTForward(J=1, wave='haar', mode='symmetric')
        outChannel_conv2 = outC2 // 2
        self.conv2_1 = nn.Conv2d(nIn_conv2, outChannel_conv2, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(outChannel_conv2)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(outChannel_conv2, outC2, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(outC2)
        self.relu2_2 = nn.ReLU()

        self.dwt3 = DWTForward(J=1, wave='haar', mode='symmetric')
        outChannel_conv3 = outC3 // 2
        nIn_conv3 = 4
        self.conv3_1 = nn.Conv2d(nIn_conv3, outChannel_conv3, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(outChannel_conv3)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(outChannel_conv3, outC3, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(outC3)
        self.relu3_2 = nn.ReLU()


    def forward(self, x):
        dwt1_1_l, dwt1_1_h = self.dwt1(x)
        dwt1_1 = torch.cat((dwt1_1_l, dwt1_1_h[0][:,:,0], dwt1_1_h[0][:,:,1], dwt1_1_h[0][:,:,2]), dim=1)
        conv1_1 = self.conv1_1(dwt1_1)
        bn1_1 = self.bn1_1(conv1_1)
        relu1_1 = self.relu1_1(bn1_1)
        conv1_2 = self.conv1_2(relu1_1)
        bn1_2 = self.bn1_2(conv1_2)
        relu1_2 = self.relu1_2(bn1_2)

        dwt2_1_l, dwt2_1_h = self.dwt2(dwt1_1_l)
        dwt2_1 = torch.cat((dwt2_1_l, dwt2_1_h[0][:, :, 0], dwt2_1_h[0][:, :, 1], dwt2_1_h[0][:, :, 2]), dim=1)
        conv2_1 = self.conv2_1(dwt2_1)
        bn2_1 = self.bn2_1(conv2_1)
        relu2_1 = self.relu2_1(bn2_1)
        conv2_2 = self.conv2_2(relu2_1)
        bn2_2 = self.bn2_2(conv2_2)
        relu2_2 = self.relu2_2(bn2_2)

        dwt3_1_l, dwt3_1_h = self.dwt3(dwt2_1_l)
        dwt3_1 = torch.cat((dwt3_1_l, dwt3_1_h[0][:, :, 0], dwt3_1_h[0][:, :, 1], dwt3_1_h[0][:, :, 2]), dim=1)
        conv3_1 = self.conv3_1(dwt3_1)
        bn3_1 = self.bn3_1(conv3_1)
        relu3_1 = self.relu3_1(bn3_1)
        conv3_2 = self.conv3_2(relu3_1)
        bn3_2 = self.bn3_2(conv3_2)
        relu3_2 = self.relu3_2(bn3_2)

        return relu1_2, relu2_2, relu3_2



class ConvLstmCell(nn.Module):
    def __init__(self, feature_size, num_features, gpu_ids, forget_bias=1,  bias=True):
        super(ConvLstmCell, self).__init__()
        self.gpu_ids = gpu_ids
        self.feature_size = feature_size
        self.num_features = num_features
        self.forget_bias = forget_bias
        self.activation = nn.Tanh()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(num_features * 2, num_features * 4, feature_size, padding=int((feature_size - 1) / 2),
                              bias=bias)

    def forward(self, input, state):

        c, h = torch.chunk(state, 2, dim=1)

        conv_input = torch.cat((input, h), dim=1)

        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            conv_output = self.conv(conv_input)
        else:
            conv_output = self.conv(conv_input)
        (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
        new_c = c * F.sigmoid(f + self.forget_bias) + F.sigmoid(i) * self.activation(j)
        new_h = self.activation(new_c) * F.sigmoid(o)
        new_state = torch.cat((new_c, new_h), dim=1)
        return new_h, new_state



def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'STMF':
        net.apply(weights_init_stmf)
    elif init_type == 'zeros':
        net.apply(weights_init_zeros)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class Decoder(nn.Module):
    def __init__(self, c_dim, gf_dim, gpu_ids):
        super(Decoder, self).__init__()
        self.gpu_ids = gpu_ids
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        deconv3_3 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 8, 3, padding=1)
        relu3_3 = nn.ReLU()
        deconv3_2 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, 3, padding=1)
        relu3_2 = nn.ReLU()
        deconv3_1 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
        relu3_1 = nn.ReLU()
        dec3 = [deconv3_3, relu3_3, deconv3_2, relu3_2, deconv3_1, relu3_1]
        self.dec3 = nn.Sequential(*dec3)

        deconv2_2 = nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, 3, padding=1)
        relu2_2 = nn.ReLU()
        deconv2_1 = nn.ConvTranspose2d(gf_dim * 2, gf_dim, 3, padding=1)
        relu2_1 = nn.ReLU()
        dec2 = [deconv2_2, relu2_2, deconv2_1, relu2_1]
        self.dec2 = nn.Sequential(*dec2)

        deconv1_2 = nn.ConvTranspose2d(gf_dim, gf_dim, 3, padding=1)
        relu1_2 = nn.ReLU()
        deconv1_1 = nn.ConvTranspose2d(gf_dim, c_dim, 3, padding=1)
        tanh1_1 = nn.Tanh()
        dec1 = [deconv1_2, relu1_2, deconv1_1, tanh1_1]
        self.dec1 = nn.Sequential(*dec1)

    def forward(self, x):
        if len(self.gpu_ids) > 0 and isinstance(x.data, torch.cuda.FloatTensor):
            input3 = fixed_unpooling(x, self.gpu_ids)
            dec3_out = self.dec3(input3)
            input2 = fixed_unpooling(dec3_out, self.gpu_ids)
            dec2_out = self.dec2(input2)
            input1 = fixed_unpooling(dec2_out, self.gpu_ids)
            dec1_out = self.dec1(input1)
            return dec1_out
        else:
            input3 = fixed_unpooling(x, self.gpu_ids)
            dec3_out = self.dec3(input3)
            input2 = fixed_unpooling(dec3_out, self.gpu_ids)
            dec2_out = self.dec2(input2)
            input1 = fixed_unpooling(dec2_out, self.gpu_ids)
            dec1_out = self.dec1(input1)
            return dec1_out

from torch.autograd import Variable

def fixed_unpooling(x, gpu_ids):
    x = x.permute(0, 2, 3, 1)
    if len(gpu_ids) > 0:

        out = torch.cat((x, Variable(torch.zeros(x.size())).cuda()), dim=3)
        out = torch.cat((out, Variable(torch.zeros(out.size())).cuda()), dim=2)
    else:
        out = torch.cat((x, Variable(torch.zeros(x.size()))), dim=3)
        out = torch.cat((out, Variable(torch.zeros(out.size()))), dim=2)

    sh = x.size()
    s0, s1, s2, s3 = int(sh[0]), int(sh[1]), int(sh[2]), int(sh[3])
    s1 *= 2
    s2 *= 2
    return out.view(s0, s1, s2, s3).permute(0, 3, 1, 2)

from torch.nn import init
def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_zeros(m):
    classname = m.__class__.__name__
    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        # init.xavier_normal(m.weight.data, gain=1)
        # init.uniform(m.weight.data, 0.0, 0.02)
        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()


def weights_init_stmf(m):
    classname = m.__class__.__name__

    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.xavier_normal_(m.weight.data, gain=1)
        # init.constant(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()


def define_discriminator(img_size, c_dim, in_num, out_num, df_dim, init_type="STMF", gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    discriminator = Discriminator(img_size, c_dim, in_num, out_num, df_dim, gpu_ids)

    if len(gpu_ids) > 0:
        discriminator.cuda(device=gpu_ids[0])

    init_weights(discriminator, init_type='zeros')
    return discriminator

class Discriminator(nn.Module):
    def __init__(self, img_size, c_dim, in_num, out_num, df_dim, gpu_ids):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        h, w = img_size[0], img_size[1]

        conv0 = nn.Conv2d(c_dim * (in_num + out_num), df_dim, 4, stride=2, padding=1)
        h = math.floor((h + 2 * 1 - 4) / 2 + 1)
        w = math.floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        lrelu0 = nn.LeakyReLU(0.2)

        conv1 = nn.Conv2d(df_dim, df_dim * 2, 4, stride=2, padding=1)
        h = math.floor((h + 2 * 1 - 4) / 2 + 1)
        w = math.floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn1 = nn.BatchNorm2d(df_dim * 2, eps=0.001, momentum=0.1)
        bn1 = nn.BatchNorm2d(df_dim * 2)
        lrelu1 = nn.LeakyReLU(0.2)

        conv2 = nn.Conv2d(df_dim * 2, df_dim * 4, 4, stride=2, padding=1)
        h = math.floor((h + 2 * 1 - 4) / 2 + 1)
        w = math.floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn2 = nn.BatchNorm2d(df_dim * 4, eps=0.001, momentum=0.1)
        bn2 = nn.BatchNorm2d(df_dim * 4)
        lrelu2 = nn.LeakyReLU(0.2)

        conv3 = nn.Conv2d(df_dim * 4, df_dim * 8, 4, stride=2, padding=1)
        h = math.floor((h + 2 * 1 - 4) / 2 + 1)
        w = math.floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn3 = nn.BatchNorm2d(df_dim * 8, eps=0.001, momentum=0.1)
        bn3 = nn.BatchNorm2d(df_dim * 8)
        lrelu3 = nn.LeakyReLU(0.2)

        D = [conv0, lrelu0, conv1, bn1, lrelu1, conv2, bn2, lrelu2, conv3, bn3, lrelu3]
        self.D = nn.Sequential(*D)

        in_features = int(h * w * df_dim * 8)

        # torch.nn.Linear(in_features, out_features, bias=True)
        self.linear = nn.Linear(in_features, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, batch_size):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            D_output = nn.parallel.data_parallel(self.D, input, self.gpu_ids)
            D_output = D_output.view(batch_size, -1)
            h = nn.parallel.data_parallel(self.linear, D_output, self.gpu_ids)
            h_sigmoid = nn.parallel.data_parallel(self.sigmoid, h, self.gpu_ids)
            return h_sigmoid, h
        else:
            D_output = self.D(input)
            D_output = D_output.view(batch_size, -1)
            h = self.linear(D_output)
            h_sigmoid = self.sigmoid(h)
            return h_sigmoid, h

def define_gdl(c_dim, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    gdl = GDL(c_dim, gpu_ids)

    if len(gpu_ids) > 0:
        gdl.cuda(device=gpu_ids[0])

    return gdl

import numpy as np
import torch.nn.functional as F

class GDL(nn.Module):
    def __init__(self, c_dim, gpu_ids):
        super(GDL, self).__init__()
        self.gpu_ids = gpu_ids
        self.loss = nn.L1Loss()
        a = np.array([[-1, 1]])
        b = np.array([[1], [-1]])
        self.filter_w = np.zeros([c_dim, c_dim, 1, 2])
        self.filter_h = np.zeros([c_dim, c_dim, 2, 1])
        for i in range(c_dim):
            self.filter_w[i, i, :, :] = a
            self.filter_h[i, i, :, :] = b

    def __call__(self, output, target):
        # pdb.set_trace()
        if len(self.gpu_ids) > 0:
            filter_w = Variable(torch.from_numpy(self.filter_w).float().cuda())
            filter_h = Variable(torch.from_numpy(self.filter_h).float().cuda())
        else:
            filter_w = Variable(torch.from_numpy(self.filter_w).float())
            filter_h = Variable(torch.from_numpy(self.filter_h).float())
        output_w = F.conv2d(output, filter_w, padding=(0, 1))
        output_h = F.conv2d(output, filter_h, padding=(1, 0))
        target_w = F.conv2d(target, filter_w, padding=(0, 1))
        target_h = F.conv2d(target, filter_h, padding=(1, 0))
        return self.loss(output_w, target_w) + self.loss(output_h, target_h)


from torch.optim import lr_scheduler
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
