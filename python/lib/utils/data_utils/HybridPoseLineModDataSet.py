import os
import sys

sys.path.append('.')
sys.path.append('..')

from config import cfg
from PIL import Image
from data_utils import read_rgb_np, read_rotation, read_translation
from torch.utils.data import Dataset
from torchvision import transforms


class HybridPoseLineModDataSet(Dataset):
    """
    HybridPose LineMod DataSetからデータを読み出すモジュール
    使用例：

    ```python
    # 'ape'のファイルを操作する場合
    linemod = HybridPoseLineModDataSet(object_name='ape')

    # 辞書配列で格納された、1つ目のデータを読み出す。
    print(linemod[0])
    ```
    """

    def __init__(self, object_name='all'):
        # directory setting
        self.base_dir = cfg.HYBRIDPOSE_LINEMOD_DIR

        self.image_shape = (480, 640)  # (h, w)

        # Use Object
        self.object_names = cfg.linemod_cls_names
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
                                     os.listdir(os.path.join(self.base_dir, object_name, 'data')))))
            self.lengths[object_name] = length
            self.total_length += length

        # pre-load-data into memory
        self.pts2d = {}
        self.pts3d = {}
        self.normals = {}
        for object_name in self.object_names:
            # keypoints
            pts2d_name = os.path.join(
                self.base_dir, 'keypoints', object_name, 'keypoints_2d.npy')

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:
            data_path = os.path.join(
                self.base_dir, object_name, 'data')  # ロードするデータのパス

            if local_idx < self.lengths[object_name]:
                # image
                image_name = os.path.join(
                    data_path, 'color{}.jpg'.format(local_idx))
                image = transforms.ToTensor()(Image.open(image_name).convert('RGB'))
                # pose
                R_name = os.path.join(
                    data_path, 'rot{}.rot'.format(local_idx)
                )
                R = read_rotation(R_name)
                # translation
                t_name = os.path.join(
                    data_path, 'tra{}.tra'.format(local_idx)
                )
                t = read_translation(t_name)

                data = {
                    'object_name': object_name,
                    'local_idx': local_idx,
                    'image_name': image_name,
                    'image': image,
                    'R': R,
                    't': t
                }

                return data

            else:
                local_idx -= self.lengths[object_name]
        raise ValueError('Invalid index: {}'.format(idx))


if __name__ == "__main__":
    ape_dataset = HybridPoseLineModDataSet(object_name='ape')
    print(ape_dataset[0])
