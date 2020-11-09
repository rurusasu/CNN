import sys

sys.path.append('.')
sys.path.append('..')

import math
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from utils.config.config import cfg

__all__ = ['ResNet', 'resnet18', 'resnet34',
           'resnet50', 'resnet108', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_channels: int,
            out_channels: int,
            stride=1,
            dilation=1):
    """3x3 convolution with padding"""

    kernel_size = np.array((3, 3))

    # 指定された拡張率でアップサンプリングされたフィルタのサイズを計算します。
    upsamled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # 出力空間サイズが入力空間サイズと等しいことを意味するフルパディングに必要なパディングを決定する。
    full_padding = (upsamled_kernel_size - 1) // 2

    #  Conv2d は numpy 配列を引数として受け付けない.
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels,
                             out_channels=out_channels,
                             stride=stride,
                             dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             stride=stride,
                             dilation=dilation)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels * 4,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.dowmsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers: list,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32):

        # 出力ストライドを追跡するために変数を追加します。
        # これは指定した出力歩幅を達成するために必要です。
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.in_channels = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
        # 最新の不安定版 torch 4.0 では、tensor.copy_method が変更され、以前のようには動作しません。
        #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:

            # すでに望ましい出力ストライドを達成しているかどうかを確認します。
            if self.current_stride == self.output_stride:

                # その場合は、現在の空間分解能を維持するために、サブサンプリングのdilationを置き換える。
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:

                # そうでない場合は、サブサンプリングを行い、現在の新しい出力ストライドを更新します。
                self.current_stride = self.current_stride * stride

            # 1x1の畳み込みはしません。
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample, dilation=self.current_dilation))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, dilation=self.current_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x = self.maxpool(x2s)

        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x32s = self.layer4(x16s)
        x = x32s

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        xfc = self.fc(x)

        return x2s, x4s, x8s, x16s, x32s, xfc



def resnet18(pretrained=False, **kwargs):
    """ResNet-18モデルを構築します。

    Args:
        pretrained (bool):
            Trueの場合, ImageNetで事前に学習されたモデルを返します．
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """ResNet-34モデルを構築します。

    Args:
        pretrained (bool):
            Trueの場合, ImageNetで事前に学習されたモデルを返します．
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """ResNet-50モデルを構築します。

    Args:
        pretrained (bool):
            Trueの場合, ImageNetで事前に学習されたモデルを返します．
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],model_dir=proj_cfg.MODEL_DIR))
    return model


def resnet101(pretrained=False, **kwargs):
    """ResNet-101モデルを構築します。

    Args:
        pretrained (bool):
            Trueの場合, ImageNetで事前に学習されたモデルを返します．
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],model_dir=proj_cfg.MODEL_DIR))
    return model


def resnet152(pretrained=False, **kwargs):
    """ResNet-152モデルを構築します。

    Args:
        pretrained (bool):
            Trueの場合, ImageNetで事前に学習されたモデルを返します．
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'],model_dir=proj_cfg.MODEL_DIR))
    return model

if __name__ == "__main__":
    #model = resnet18()
    model = resnet18(pretrained=True)
    #model = resnet34()
    #model = resnet50()
    #model = resnet101()
    #model = resnet152()
    print('Road Complete!')