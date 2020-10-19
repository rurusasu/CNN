import os
import sys

sys.path.append('.')
sys.path.append('..')

from utils import save_Excel
from config.config import cfg
from data_utils import read_rgb_np, read_rotation, read_translation
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class LineModDataSet(Dataset):
    """
    LineMod DataSet を操作するモジュール。
    使用例：

    ```python
    # 'ape'のファイルを操作する場合
    linemod = LineModDataSet(object_name='ape')

    # 辞書配列で格納された、1つ目のデータを読み出す。
    print(linemod[0])
    ```
    """

    def __init__(self, object_name='all'):
        # directory setting
        self.base_dir = cfg.LINEMOD_DIR

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

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:
            data_path = os.path.join(
                self.base_dir, object_name, 'data')  # ロードするデータのパス

            if local_idx < self.lengths[object_name]:
                # image
                image_name = os.path.join(
                    data_path, 'color{}.jpg'.format(local_idx))
                #image = transforms.ToTensor()(Image.open(image_name).convert('RGB'))
                image = read_rgb_np(image_name)
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

    def save_data(self, data: dict):
        for object_name in self.object_names:
            save_path = os.path.join(self.base_dir, object_name, '{}.xlsx').format(
                object_name)  # データを保存する場所のパス

            if isinstance(data, dict):
                if 'R' in data.keys():
                    R = data['R']

                    R_1 = R[0]
                    R_2 = R[1]
                    R_3 = R[2]

                    R = {
                        'R_1_x': R_1[0], 'R_1_y': R_1[1], 'R_1_z': R_1[2],
                        'R_2_x': R_2[0], 'R_2_y': R_2[1], 'R_2_z': R_2[2],
                        'R_3_x': R_3[0], 'R_3_y': R_3[1], 'R_3_z': R_3[2]
                    }

                    # R の値を、変換後の値で置き換える。
                    del data['R']  # key 'R' の要素を削除
                    data.update(R)

                if 't' in data.keys():
                    t = data['t']
                    t_x = t[0]
                    t_y = t[1]
                    t_z = t[2]

                    t = {'t_x': t_x[0], 't_y': t_y[0], 't_z': t_z[0]}

                    # t の値を、変換後の値で置き換える。
                    del data['t']
                    data.update(t)

                save_Excel(save_path, data, img_past=True)

            else:
                raise ValueError('Invalid Data Format!')


if __name__ == "__main__":
    ape_dataset = LineModDataSet(object_name='ape')
    print(ape_dataset[0])
