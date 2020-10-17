import os
import sys
sys.path.append('.')
sys.path.append('..')

from config import cfg
from utils import save_Excel
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob



def read_rgb_np(rgb_path):

    # PIL は極端に大きな画像など高速にロードできない画像は見過ごす仕様になっている。
    # `LOAD_TRUNCATED_IMAGES` を `True` に設定することで、きちんとロードされるようになる。
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert("RGB")
    img = np.array(img, np.uint8)
    return img


def initializer(self, base_dir, All_object_names, object_name='all'):
    # directory setting
    self.base_dir = base_dir

    self.image_shape = (480, 640)  # (h, w)

    # Use Object
    self.object_names = All_object_names
    if object_name is 'all':
        pass
    elif object_name in self.object_names:
        self.object_names = [object_name]
    else:
        raise ValueError('Invaild object name: {}' .format(object_name))

    self.lengths = {}
    self.total_length = 0

    return self


def read_rotation(filename):
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


def read_translation(filename):
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


class LineModDataSet(Dataset):
    """LineMod DataSetを操作するためのモジュール

    モジュール内のデータは全て Pandas DataFrame に統一する。

    """
    def __init__(self, object_name='all'):
        self = initializer(self,
                           base_dir=cfg.LINEMOD_DIR,
                           All_object_names=cfg.linemod_cls_names, object_name=object_name)

        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endswith('jpg'),
                                     os.listdir(os.path.join(self.base_dir, object_name, 'data')))))
            self.lengths[object_name] = length
            self.total_length += length


    def __setitem__(self):
        """
        """

        """与えられたベースディレクトリ内にあるディレクトリ名を読み出す関数
        """
        dir_pathes = [f for f in base_dir if os.path.isdir(os.path.join(base_dir, f))]

        # dir_pathes の image ディレクトリから画像パスを読み込む
        for dir_path in dir_pathes:
            print('{} のimage ディレクトリから画像パスを読み出します。'.format(dir_path))

            img_paths = os.path.join(cfg.LINEMOD_ORIG_DIR, 'ape'+ os.sep + 'image' + os.sep + '*.{}'.format('jpg')) 
            img_paths = glob.glob(img_paths, recursive=True)
            
            for img_path in img_paths:
                img = read_rgb_np(img_path)
                


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


class HybridPose_LineModDataSet(Dataset):
    def __init__(self, object_name='all'):
        self = initializer(self,
                           base_dir=cfg.HYBRIDPOSE_LINEMOD_DIR,
                           All_object_names=cfg.linemod_cls_names, object_name=object_name)

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
            #keypoints
            pts2d_name = os.path.join(self.base_dir, 'keypoints', object_name, 'keypoints_2d.npy')

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


class PVNet_LineModDataSet(Dataset):
    def __init__(self, object_name='all'):
        self = initializer(self,
                           base_dir=cfg.PVNet_LINEMOD_DIR,
                           All_object_names=cfg.linemod_cls_names,
                           object_name=object_name)

        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endswith('jpg'),
                                     os.listdir(os.path.join(self.base_dir, object_name, 'JPEGImages')))))
            self.lengths[object_name] = length
            self.total_length += length


class OcclusionLineModDataSet(Dataset):
    def __init__(self, object_name='all'):
        self = initializer(self,
                           base_dir=cfg.OCCLUSION_LINEMOD_DIR,
                           All_object_names=cfg.occ_linemod_cls_names,
                           object_name=object_name)

        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endwidth('txt'),
                                     os.listdir(os.path.join(self.base_dir, 'poses', object_name)))))
            self.length[object_name] = length
            self.total_length += length


def matrix(data):

    x = data[0]
    y = data[1]
    z = data[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    """
    ax.scatter(np.ravel(data[0]),  # x軸
               np.ravel(data[1]),  # y軸
               np.ravel(data[2]),  # z軸
               s=5,  # マーカーの大きさ
               )
    """

    ax.scatter(x[0], y[0], z[0], s=20, marker=".")
    ax.scatter(x[1], y[1], z[1], s=20, marker="^")
    ax.scatter(x[2], y[2], z[2], s=20, marker="s")
    ax.scatter(x[3], y[3], z[3], s=20, marker="4")

    plt.show()


if __name__ == "__main__":
    import glob

    #base_dir = os.path.join(cfg.LINEMOD_ORIG_DIR, '**' + os.sep)
    base_dir = os.listdir(cfg.LINEMOD_ORIG_DIR)
    print(base_dir)

    #print([os.path.basename(p.rstrip(os.sep)) for p in glob.glob(base_dir, recursive=True)])
    #print([f for f in base_dir if os.path.isdir(os.path.join(cfg.LINEMOD_ORIG_DIR, f))])
    
    pth = os.path.join(cfg.LINEMOD_ORIG_DIR, 'ape'+ os.sep + '**'+os.sep + '*.{}'.format('jpg')) 

    img_path = glob.glob(pth, recursive=True)
    print(img_path)
    # linemod = LineModModelDB()
    # result = linemod.get_corners_3d(cfg.linemod_cls_names[0])

    #ape_dataset = LineModDataSet(object_name='ape')
    #result = ape_dataset.__getitem__(1)
    #ape_dataset.save_data(result)

    """
    data = np.array([[-0.113309, 0.991361, -0.0660649],
                     [-0.585432, -0.0128929, 0.810619],
                     [0.802764, 0.130527, 0.581836],
                     [-0.13079199, -0.0783575, 1.04177]])
    data = data.T

    matrix(data)
    """
