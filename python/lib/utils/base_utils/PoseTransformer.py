import os
import sys

sys.path.append('.')
sys.path.append('..')

from data_utils.data_utils import load_ply_model
from config.config import cfg
import numpy as np


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])

    blender_models = {}
    translation_transforms = {}
    # Occlusion linemod の models ディレクトリに含まれる xyz データのナンバー
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }

    def __init__(self, object_name):
        self.object_name = object_name
        self.blender_model_path = os.path.join(
            cfg.PVNET_LINEMOD_DIR, '{}/{}.ply'.format(object_name, object_name))
        self.orig_model_path = os.path.join(
            cfg.LINEMOD_ORIG_DIR, '{}/mesh.ply'.format(object_name))
        self.xyz_pattern = os.path.join(
            cfg.OCCLUSION_LINEMOD_DIR, 'models/{}/{}.xyz')
        #self.model_aligner = ModelAligner(object_name)

    def get_blender_model(self):
        """Blender モデルを返す関数

        Return
        ------
        array (ndarray):
            n行3列の array型 行列
        """

        if self.object_name in self.blender_models:
            return self.blender_models[self.object_name]

        blender_model = load_ply_model(
            self.blender_model_path.format(self.object_name, self.object_name))
        self.blender_models[self.object_name] = blender_model

        return blender_model

    def get_translation_transform(self):
        """Occlusion linemod データセットに txt 形式で保存されているモデルの座標データと
        Blender で作ったモデルの座標データのそれぞれの平均値の差を translation_transforms として保存する関数

        Return
        ------
        translation_transform (list):
            1行3列の list
        """

        if self.object_name in self.translation_transforms:
            return self.translation_transforms[self.object_name]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(
            self.object_name.title(), self.class_type_to_number[self.object_name]))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 1.],
                             [0., 1., 0.]])
        xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        self.translation_transforms[self.object_name] = translation_transform

        return translation_transform

    def orig_pose_to_blender_pose(self, pose):
        """LineMod オリジナルの姿勢を Blender の姿勢に変換する関数

        Param
        -----
        pose ():

        """
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

    @staticmethod
    def blender_pose_to_blender_euler(pose):
        """Blender の姿勢を Blender の euler 姿勢へ変換する関数
        """
        euler = [r / np.pi * 180 for r in mat2euler(pose, axes='szxz')]
        euler[0] = -(euler[0] + 90) % 360
        euler[1] = euler[1] - 90
        return np.array(euler)

    def orig_pose_to_blender_euler(self, pose):
        blender_pose = self.orig_pose_to_blender_pose(pose)
        return self.blender_pose_to_blender_euler(blender_pose)

    def occlusion_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                            [0., 0., 1.],
                            [1., 0., 0.]])
        rot = np.dot(rot, rotation)

        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform())
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose


if __name__ == "__main__":
    object_name = 'ape'

    pose_transformer = PoseTransformer(object_name)

    # Get blender model test
    # blender_model = pose_transformer.get_blender_model()
    # print(blender_model)

    # Get translation transform test
    translation_transform = pose_transformer.get_translation_transform()
    print(translation_transform)