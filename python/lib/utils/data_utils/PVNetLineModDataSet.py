import os
import sys

sys.path.append('.')
sys.path.append('..')


import json
from config.config import cfg
from data_utils import read_rgb_np, read_mask_np
from torchvision import transforms
from torch.utils.data import Dataset


class PVNetLineModDataSet(Dataset):
    """PVNet で使用された LineMod Dataset を操作するモジュール。

    使用例：

    ```python
    # 'ape'のファイルを操作する場合
    linemod = PVNetLineModDataSet(object_name='ape')

    # 辞書配列で格納された、1つ目のデータを読み出す。
    print(linemod[0])
    ```
    """

    def __init__(self, object_name='all'):
        self.base_dir = cfg.PVNET_LINEMOD_DIR  # directory setting
        self.image_shape = (480, 640)  # (h, w)
        self.object_names = cfg.linemod_cls_names  # Use Object
        if object_name is 'all':
            pass
        elif object_name in self.object_names:
            self.object_names = [object_name]
        else:
            raise ValueError('Invaild object name: {}' .format(object_name))

        self.lengths = {}
        self.total_length = 0

        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endswith('jpg'),
                                     os.listdir(os.path.join(self.base_dir, object_name, 'JPEGImages')))))
            self.lengths[object_name] = length
            self.total_length += length

        with open(cfg.linemod_config, 'r') as f:
            config = json.load(f)
        self.cfg = config
        self.img_transforms = transforms.Compose([
            transforms.ColorJitter(
                self.cfg['brightness'], self.cfg['contrast'], self.cfg['saturation'], self.cfg['hue']),
            transforms.ToTensor(),  # image の dtypeが np.uint8 の場合, 255 で除算されます.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms = transforms.Compose([
            transforms.ToTensor(),  # image の dtypeが np.uint8 の場合, 255 で除算されます.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int):
        local_idx = idx
        for object_name in self.object_names:
            data_path = os.path.join(self.base_dir, object_name)
            image_dir_path = os.path.join(data_path, 'JPEGImages')
            mask_dir_path = os.path.join(data_path, 'mask')

            if local_idx < self.lengths[object_name]:
                image_name = os.path.join(image_dir_path,
                                          # 1.jpg → 000001.jpg
                                          '{}.jpg'.format(str(local_idx).zfill(6)))
                image = read_rgb_np(image_name)
                mask_name = os.path.join(mask_dir_path,
                                         # 1.png → 0001.png
                                         '{}.png'.format(str(local_idx).zfill(4)))
                mask = read_mask_np(mask_name)

                data = {
                    'object_name': object_name,
                    'local_idx': local_idx,
                    'image_name': image_name,
                    'image': image,
                    'mask_name': mask_name,
                    'mask': mask
                }

                return data

            else:
                local_idx -= self.length[object_name]
        raise ValueError('Invalid index: {}'.format(idx))


if __name__ == "__main__":
    pvlinemod = PVNetLineModDataSet('ape')
    print(pvlinemod[1])
