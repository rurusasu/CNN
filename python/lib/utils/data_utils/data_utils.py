import os
import sys
sys.path.append('.')
sys.path.append('..')

from utils import read_pickle
from config.config import cfg
import torch
import numpy as np
from torch.utils.data import Dataset
from plyfile import PlyData
from PIL import Image, ImageFile


def load_ply_model(model_path: str):
    """ply データに格納されたモデルの x, y, z 座標を array 配列で読み出す関数

    Param
    -----
    model_path (str):
        ply データのパス

    Return
    ------
    array (ndarray):
        n行3列の array型 行列
    """

    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)


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


def read_transform_dat(transform_dat_path: str):
    """dat 形式のファイルを読み込む関数

    Param
    -----
    transform_dat_path (str):
        dat ファイルへのパス

    Return
    ------
    transform_dat (list):
        要素に numpy 配列を持った リスト
    """

    transform_dat = np.loadtxt(transform_dat_path, skiprows=1)[:, 1]
    transform_dat = np.reshape(transform_dat, newshape=[3, 4])
    return transform_dat


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
    from config.config import cfg

    pv_linemod_path = cfg.PVNET_LINEMOD_DIR
    object_name = 'ape'
    base_dir = os.path.join(pv_linemod_path, object_name)

    """
    # load_ply_model test
    model_path = os.path.join(base_dir, '{}.ply'.format(object_name))
    plydata = load_ply_model(model_path)
    print(plydata)
    """

    # read transform dat test
    dat_path = os.path.join(cfg.LINEMOD_ORIG_DIR,
                            '{}/transform.dat'.format(object_name))
    transform_dat = read_transform_dat(dat_path)
    print(transform_dat)

    """
    data = np.array([[-0.113309, 0.991361, -0.0660649],
                     [-0.585432, -0.0128929, 0.810619],
                     [0.802764, 0.130527, 0.581836],
                     [-0.13079199, -0.0783575, 1.04177]])
    data = data.T

    matrix(data)
    """
