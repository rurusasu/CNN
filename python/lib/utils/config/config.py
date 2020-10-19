from easydict import EasyDict
import os
import sys

cfg = EasyDict()

"""
Path setting
"""

cfg.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.UTILS_DIR = os.path.dirname(cfg.CONFIG_DIR)
cfg.LIB_DIR = os.path.dirname(cfg.UTILS_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')

cfg.DATASETS_DIR = os.path.join(cfg.LIB_DIR, 'datasets')
cfg.NETWORKS_DIR = os.path.join(cfg.LIB_DIR, 'networks')

cfg.MODELS_DIR = os.path.join(cfg.DATA_DIR, 'models')
cfg.REC_DIR = os.path.join(cfg.DATA_DIR, 'record')

cfg.DATAUTILS_DIR = os.path.join(cfg.UTILS_DIR, 'data_utils')


def add_path():
    """システムのファイルパスを設定するための関数
    """

    for key, value in cfg.items():
        if 'DIR' in key:
            sys.path.insert(0, value)


add_path()


"""
Data path settings
"""
cfg.LINEMOD_DIR = os.path.join(cfg.DATA_DIR, 'linemod')
cfg.linemod_config = os.path.join(cfg.CONFIG_DIR, 'default_linemod_cfg.json')
cfg.LINEMOD_ORIG_DIR = os.path.join(cfg.DATA_DIR, 'linemod_orig')
cfg.OCCLUSION_LINEMOD_DIR = os.path.join(cfg.DATA_DIR, 'OCCLUSION_LINEMOD')
cfg.HYBRIDPOSE_LINEMOD_DIR = os.path.join(cfg.DATA_DIR, 'HybridPose_linemod')
cfg.PVNET_LINEMOD_DIR = os.path.join(cfg.DATA_DIR, 'PVNet_linemod')
cfg.YCB_DIR = os.path.join(cfg.DATA_DIR, 'YCB')

"""
Class Name
"""
cfg.linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone',
                         'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']

cfg.occ_linemod_cls_names = ['ape', 'can', 'cat',
                             'driller', 'duck', 'eggbox', 'glue', 'holepuncher']

if __name__ == "__main__":
    add_path()
