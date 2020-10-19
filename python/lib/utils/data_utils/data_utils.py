import os
import sys
sys.path.append('.')
sys.path.append('..')

import glob
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plyfile import PlyData
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from utils import read_pickle, save_Excel
from config.config import cfg



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


def read_mask(mask_path: str):
    """mask 画像を読み込み torch の long 型に変換する関数

    Param
    -----
    mask_path (str):
        読み込む mask 画像のパス

    Return
    ------

    """

    mask = Image.open(mask_path).convert('1')  # 1 bit で画像を読み込み
    mask_seg = np.array(mask).astype(np.int32)
    return torch.from_numpy(mask_seg).long()


def read_mask_np(mask_path: str):
    """mask 画像を読み込み ndarray に変換する関数

    Param
    -----
    mask_path (str):
        読み込む mask 画像のパス

    Return
    ------

    """

    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg


def read_rgb(rgb_path: str):
    """画像を読み込み torch の tensor に変換する関数

    Param
    -----
    rgb_path (str):
        読み込む画像のパス

    Return
    ------
    torch_tensor (tensor):
        float 32 のテンソル
    """

    # PIL は極端に大きな画像など高速にロードできない画像は見過ごす仕様になっている。
    # `LOAD_TRUNCATED_IMAGES` を `True` に設定することで、きちんとロードされるようになる。
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert("RGB")
    img = np.array(img)
    return torch.from_numpy(img).float().permute(2, 0, 1)


def read_rgb_np(rgb_path: str):
    """画像を読み込み ndarray に変換する関数

    Param
    -----
    rgb_path (str):
        読み込む画像のパス

    Return
    ------
    img (ndarray):
        uint 8 の numpy 配列
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    return img


def read_rotation(filename):
    """object の回転角度を読み込む関数

    Param
    -----
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

    Param
    -----
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


def read_vertex(vertex_path: str):
    """vertex データを pickle ファイルからロードする関数

    Param
    -----
    vertex_path (str):
        読み込む pickle ファイルのパス

    Return
    ------
    vertex:
        pickle ファイルから読み込んだ vertex データ
    """
    vertex = read_pickle(vertex_path)
    return vertex


def read_pose(pose_path):
    pose = read_pickle(pose_path)['RT']
    return pose


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
    """
    import glob
    import pandas as pd
    #base_dir = os.path.join(cfg.LINEMOD_ORIG_DIR, '**' + os.sep)
    base_dir = os.listdir(cfg.LINEMOD_ORIG_DIR)
    # print(base_dir)

    #print([os.path.basename(p.rstrip(os.sep)) for p in glob.glob(base_dir, recursive=True)])
    #print([f for f in base_dir if os.path.isdir(os.path.join(cfg.LINEMOD_ORIG_DIR, f))])

    data = {}

    img_paths = os.path.join(
        cfg.LINEMOD_ORIG_DIR, 'ape' + os.sep + '**'+os.sep + '*.{}'.format('jpg'))
    img_paths = glob.glob(img_paths, recursive=True)
    print(img_paths)

    for img_path in img_paths:
        img = read_rgb_np(img_path)
        data[img_path] = img
    """
    #df = pd.DataFrame()

    # linemod = LineModModelDB()
    # result = linemod.get_corners_3d(cfg.linemod_cls_names[0])

    ape_dataset = LineModDataSet(object_name='ape')
    print(ape_dataset[0])
    #result = ape_dataset.__getitem__(1)
    # ape_dataset.save_data(result)

    """
    data = np.array([[-0.113309, 0.991361, -0.0660649],
                     [-0.585432, -0.0128929, 0.810619],
                     [0.802764, 0.130527, 0.581836],
                     [-0.13079199, -0.0783575, 1.04177]])
    data = data.T

    matrix(data)
    """
