import os
import pandas as pd
import numpy as np

import openpyxl
from openpyxl.drawing.image import Image


def image_past(worksheet, cell_name: str, img_path: str):
    """Excel に画像を貼りつける関数

    Parames
    -------
    worksheet (openpyxl.Worksheet):
        画像を貼り付け先となる Excel の worksheet
    cell_name (str):
        画像を貼り付ける cell の番号
        例 "C2"
    img_path (str):
        貼り付ける画像をロードするためのパス
    """

    try:
        # 画像ファイルのロード
        img = Image(img_path)
        # 画像を貼り付ける
        worksheet.add_image(img, cell_name)
    except FileNotFoundError:
        # 画像が見つからない場合はスキップ
        pass


def image_past_AllColumn(worksheet, past_column: str):
    """Excel の縦列に画像を貼り付ける関数

    Parames
    -------
    worksheet (openpyxl.Worksheet):
        画像を貼り付け先となる Excel の worksheet
    past_column (str):
        画像を貼り付ける列名
    """

    # 列数を取得
    row_max = len(list(worksheet.rows))
    # 画像を貼り付け
    for i in range(1, row_max):
        idx = i + 1
        cell_name = past_column + str(idx)  # "C"+ "1" → "C1"
        img_path = worksheet[cell_name].value

        image_past(worksheet, cell_name, img_path)


def save_Excel(path: str, data: dict, img_past: bool = False):
    """辞書型のデータを Excel ファイル年て保存する関数

    Params
    ------
    path (str):
        保存する Excel ファイルへのパス
    data (dict):
        保存するデータ
    img_past (bool):
        True: 画像をセル内の画像パスと置き換える。
        False: 画像を画像パスと置き換えない。
    """

    # 辞書の値を list 型に変換
    values = [data.values()]
    # pandas DataFrame にデータを変換
    # 最初の列に key を表示
    df = pd.DataFrame(values, columns=data.keys())
    # .xlsx ファイルとして保存
    df.to_excel(path, index=False)

    if img_past:
        # Excel のロード
        wb = openpyxl.load_workbook(path)
        # Active シート取得
        ws = wb.active
        print(type(ws))
        # 画像パスの cell を画像に置き換える
        # 画像パスが入っている cell の列を取得
        # そのために、最初の行を走査する。
        col_max = ws.max_column  # 列数の最大値を取得

        for i in range(1, col_max):
            cell_value = ws.cell(column=i, row=1).value
            if 'image_name' in cell_value:
                past_column = openpyxl.utils.get_column_letter(
                    ws.cell(column=i, row=1).column)

                image_past_AllColumn(ws, past_column)

        wb.save(path)


class ModelAligner(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    # 固有の行列
    intrisic_matrix = {
        # LineMod の場合は Kinect のカメラパラメータ
        'linemod': np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self, object_name='cat'):
        self.object_name = object_name
        self.blender_model_path = os.path.join(
            cfg.PVNET_LINEMOD_DIR, '{}/{}.ply'.format(object_name, object_name))
        self.orig_model_path = os.path.join(
            cfg.LINEMOD_ORIG_DIR, '{}/mesh.ply'.format(object_name))
        self.orig_old_model_path = os.path.join(
            cfg.LINEMOD_ORIG_DIR, '{}/OLDmesh.ply'.format(object_name))
        self.transform_dat_path = os.path.join(
            cfg.LINEMOD_ORIG, '{}/transform.dat'.format(object_name))

        self.R_p2w, self.t_p2w, self.s_p2w = seld.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                               [-0.986692547798, -0.124983474612,
                                   0.104004733264, -0.050683632493],
                               [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                               [-0.196746662259, 0.978194475174,
                                   0.066531419754, 0.000077120210],
                               [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}

    def __init__(self, object_name):
        self.object_name = object_name
        self.orig_model_path = os.path.join(
            cfg.LINEMOD_ORIG, '{}/mesh.ply'.format(class_type))
        self.xyz_pattern = os.path.join(
            cfg.OCCLUTION_LINEMOD, 'models/{}/{}.xyz')
        self.model_aligner = ModelAligner()


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    data = {
        'object_name': 'ape',
        'local_idx': 1,
        'image_name': 'color1.jpg',
    }

    save_Excel(path, data)
