import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import PIL.Image
from utils.utils import save_Excel
from utils.config import cfg
from torch.utils.data import Dataset
from torchvision import transforms



# Hybrid Poseのプログラムを参考に記述
class LinemodDataset(Dataset):
    def __init__(self, object_name='all'):

        # directory settings
        self.object_names = cfg.linemod_cls_names
        self.base_dir = cfg.LINEMOD_DIR

        self.img_shape = (480, 640)  # (h, w)

        if object_name == 'all':
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

        # pre-load data into memory
        self.pts2d = {}
        self.pts3d = {}

    def read_3d_points(self, filename):
        with open(filename) as f:
            in_vertex_list = False
            vertices = []  # 頂点位置を格納する変数
            in_mm = False
            for line in F:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    vertex = np.array([[float(vertex[0])],
                                       [float(vertex[1])],
                                       [float(vertex[2])]],
                                      dtype=np.float32)

                    # 単位系を変換
                    if in_mm:
                        vertex = vertex / np.float32(10)  # mm → cm
                    vertex = vertex / np.float(100)  # cm → m
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True
        return vertices

    def read_rotation(self, filename):
        """object の回転角度を読み込む関数

        Parame
        ------
        filename (str):
            読み込むファイル名

        Return
        ------
        R (ndarray):
            object の姿勢の np 配列
        """

        with open(filename) as f:
            f.readline()
            R = []
            for line in f:
                R.append(line.split())
            R = np.array(R, dtype=np.float32)
        return R

    def read_translation(self, filename):
        """object の並進移動を読み込む関数

        Parame
        ------
        filename (str):
            読み込むファイル名

        Return
        ------
        T (ndarray):
            object の並進移動の np 配列
        """

        with open(filename) as f:
            f.readline()
            T = []
            for line in f:
                T.append([line.split()[0]])
            T = np.array(T, dtype=np.float32)
            T = T / np.float32(100)

        return T

    def read_normal(self, filename):
        """
        """

        with open(filename) as f:
            lines = f.readlines()
            normal = np.array(lines[3].strip().split(), dtype=np.float32)

        return normal

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:

            save_path = os.path.join(
                self.base_dir, object_name)  # ロードしたデータを保存する場所のパス
            data_path = os.path.join(save_path, 'data')  # ロードするデータのパス

            if local_idx < self.lengths[object_name]:
                # image
                image_name = os.path.join(
                    data_path, 'color{}.jpg'.format(local_idx))
                image = transforms.ToTensor()(PIL.Image.open(image_name).convert('RGB'))
                # keypoints
                #pts2d = self.pts2d[object_name][local_idx]
                #pts3d = self.pts3d[object_name]
                # orientation
                R_name = os.path.join(
                    data_path, 'rot{}.rot'.format(local_idx)
                )
                R = self.read_rotation(R_name)
                # translation
                t_name = os.path.join(
                    data_path, 'tra{}.tra'.format(local_idx)
                )
                t = self.read_translation(t_name)

                data = {
                    'object_name': object_name,
                    'local_idx': local_idx,
                    'image_name': image_name,
                    'image': image,
                    #'pts2d': pts2d,
                    #'pts3d': pts3d,
                    'R': R,
                    't': t
                }

                save_Excel(save_path, data)

                return data

            else:
                local_idx -= self.lengths[object_name]
        raise ValueError('Invalid index: {}'.format(idx))


if __name__ == "__main__":
    ape_dataset = LinemodDataset(object_name='ape')
    ape_dataset.__getitem__(1)
