from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch


def make_model(args, parent=False):
    return ESASN(args)

class SALayer(nn.Module):
    def __init__(self, channel, act, groups=4):
        super(SALayer, self).__init__()
        self.channel_in = channel
        self.conv_y = nn.Conv2d(in_channels=channel, out_channels=channel//4, kernel_size=1, bias=False, groups=groups)
        self.conv_z = nn.Conv2d(in_channels=channel, out_channels=channel//4, kernel_size=1, bias=False, groups=groups)
        self.act = act
        self.conv = nn.Conv2d(in_channels=channel//4, out_channels=channel, kernel_size=1, bias=False, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv_y(x)
        z = self.conv_z(x)
        temp = self.act(y*z)
        att = self.conv(temp)
        att = self.sigmoid(att)
        out = x * att + x
        return out


class ESpindleBlock(nn.Module):
    def __init__(
            self, n_feat, w_feat, kernel_size,
            bias=True, act=nn.ReLU(True), groups=4):
        super(ESpindleBlock, self).__init__()
        self.w_feat = w_feat
        
        self.widen = nn.Conv2d(n_feat, w_feat, 1, bias=bias)
        self.b1 = nn.Conv2d(w_feat//4, w_feat//4, kernel_size, padding=kernel_size//2, bias=bias, groups=w_feat//4)

        b2 = []
        b2.append(nn.Conv2d(w_feat//4, w_feat//4, kernel_size, padding=kernel_size//2, bias=bias, groups=w_feat//4))
        b2.append(act)
        self.b2 = nn.Sequential(*b2)

        b3 = []
        b3.append(nn.Conv2d(w_feat//4, w_feat//4, kernel_size, padding=kernel_size//2, bias=bias, groups=w_feat//4))
        b3.append(act)
        self.b3 = nn.Sequential(*b3)

        b4 = []
        b4.append(nn.Conv2d(w_feat//4, w_feat//4, kernel_size, padding=kernel_size//2, bias=bias, groups=w_feat//4))
        b4.append(act)
        self.b4 = nn.Sequential(*b4)
        self.sa = SALayer(w_feat, act, groups)
        self.shrink = nn.Conv2d(w_feat, n_feat, 1, bias=bias)

    def forward(self, x):
        res = self.widen(x)
        y1 = self.b1(res[:, 0:self.w_feat//4, :, :])
        y2 = self.b2(res[:, self.w_feat//4:self.w_feat//2, :, :])
        y3 = self.b3(res[:, self.w_feat//2:3*self.w_feat//4, :, :]+y2)
        y4 = self.b4(res[:, 3*self.w_feat//4:self.w_feat, :, :]+y3)

        out = self.sa(torch.cat((y1, y2, y3, y4), 1))
        out = self.shrink(out)
        out += x
        return out

class SSAG(nn.Module):
    def __init__(self, n_feat, w_feat, kernel_size, n_blocks, act=nn.ReLU(True), groups=8):
        super(SSAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [ESpindleBlock(n_feat, w_feat, 3, act=act, groups=groups) for _ in range(n_blocks)]
        self.body = nn.Sequential(*modules_body)
        self.ff = nn.Conv2d(n_blocks*n_feat, n_feat, 1, bias=False)

    def forward(self, x):
        body_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            body_out.append(x)
        res = self.ff(torch.cat(body_out, 1))
        return res


class ESASN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ESASN, self).__init__()

        self.B = args.n_resgroups
        n_blocks = args.n_resblocks
        n_feats = args.n_feats
        w_feats = args.w_feats
        groups = args.groups
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=(kernel_size // 2))]

        # define body module
        m_body = [
            SSAG(n_feats, w_feats, kernel_size, n_blocks, act=act, groups=groups)
            for _ in range(self.B)]

        # define tail module
        m_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size // 2))]

        # out_feats = self.scale * self.scale * args.n_colors
        # skip = []
        # skip.append(
        #     nn.Conv2d(args.n_colors, out_feats, 3, padding=3//2)
        # )
        # skip.append(nn.PixelShuffle(self.scale))

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.gff = nn.Conv2d(self.B*n_feats, n_feats, 1, bias=False)
        self.tail = nn.Sequential(*m_tail)
        # self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = self.sub_mean(x)
        # s = self.skip(x)
        x = self.head(x)
        
        body_out = []
        for i in range(self.B):
            if i == 0:
               temp = self.body[i](x)
            else:
               temp = self.body[i](temp)
            body_out.append(temp)
        res = self.gff(torch.cat(body_out, 1))

        res += x
        x = self.tail(res)
        # x += s
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
