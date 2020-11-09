
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


def train(net, optimizer, dataloader, epoch):
    for rec in recs: rec.reset()
    data_time.reset()
    batch_time.reset()

    train_begin=time.time()

    net.train()
    size = len(dataloader)
    end = time.time()
    for idx, data in  enumerate(dataloader):
        image, mask, vertex, vertex_weights, pose, _ = [d.cuda() for d in data]
        data_time.update(time.time()-end)

        seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights)
        loss_seg, loss_vertex, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, precision, recall)]
        loss = loss_seg + loss_vertex, train_cfg['vertex_loss_ratio']
        vals = (loss_sef, loss_vertex, precision, recall)
        for rec, val in zip(recs, val): rec.update(val)

        optimizer.zero_grad()
        loss.batkward()
        optimizer.step()

        batch_time.update(time.time()-end)
        end = time.time()

        if idx % train_cfg['loss_rec_step'] == 0:
            step = epoch * size + idx
            losses_batch=OderedDict()
            for name, rec in zip(recs_names, recs):
                losses_batch['train/'+name]=rec.avg
            encorder.rec_loss_batch(losses_batch, step, epoch)
            for rec in recs: rec.reset()

            data_time.reset()
            batch_time.reset()

        if idx % train_cfg['img_rec_step'] == 0:
            batch_size=image.shape[0]
            nrow = 5 if batch_size > 5 else batch_size
            recorder.rec_segmentation(F.softmax)
