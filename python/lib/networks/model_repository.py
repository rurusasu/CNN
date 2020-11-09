import sys

sys.path.append('.')
sys.path.append('..')

import torch

from networks.resnet import resnet18
from torch import nn
from torch.nn import functional as F


class Resnet18_8s(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=128, s8dim=64, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18_8s, self).__init__()

        # 学習済みの重みをロードし、Averageプールを削除します。
        # レイヤーの出力ストライドを 8 に設定
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim

        # 1x1 Conv スコアリング層をランダムに初期化します。
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(in_channels=resnet18_8s.in_channels,
                      out_channels=fcdim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s -> 128
        self.conv8s = nn.Sequential(
            nn.Conv2d(in_channels=128+fcdim,
                      out_channels=s8dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s -> 64
        self.conv4s = nn.Sequential(
            nn.Conv2d(in_channels=64+s8dim,
                      out_channels=s4dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s -> 64
        self.conv2s = nn.Sequential(
            nn.Conv2d(in_channels=64+s4dim,
                      out_channels=s2dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(in_channels=3+s2dim,
                      out_channels=raw_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(in_channels=raw_dim,
                      out_channels=seg_dim+ver_dim,
                      kernel_size=1,
                      stride=1)
        )

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x = self.convraw(torch.cat([fm, x], 1))
        seg_pred = x[:, :self.seg_dim, :, :]
        ver_pred = x[:, self.seg_dim:, :, :]

        return seg_pred, ver_pred

if __name__ == "__main__":
    # 入力サイズを変えてテスト
    import numpy as np
    for k in range(50):
        hi, wi = np.random.randint(0, 29), np.random.randint(0, 49)
        h, w = 256+hi*8, 256+wi*8
        print(h, w)
        img = np.random.uniform(-1, 1, [1, 3, h, w]).astype(np.float32)
        net = Resnet18_8s(2, 2).cuda()
        out = net(torch.tensor(img).cuda())