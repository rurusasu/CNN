
import numpy as np
from torch import nn

from lib.utils.arg_utils import parse_args
from lib.datasets.linemod import LinemodDataset

def initialize(args):
    """学習条件の初期設定を行う関数
    """
    np.random.seed(0)


def setup_loaders(args):
    if args.dataset == 'linemod':
        full_set = LinemodDataset(object_name=args.object_name)

    # 学習
    train_size = int(0.8 * len(fullset))

class NetWrapper(nn.Model):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, image):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        