import sys

sys.path.append('.')
sys.path.append('..')

from config import cfg
import os
import numpy as np


class evaluate(object):
    def __init__(self):
        self.mesh = os.path.join(cfg.LINEMOD_DIR, '{}/mesh.ply')

    def read_3d_points_linemod(self, object_name):
        filename = self.mesh.format(object_name)
        with open(filename) as f:
            in_vertex_list = False
            vertices = []
            in_mm = False
            for line in f:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    # x, y, z 軸座標ごとの値を読み込む
                    vertex = np.array([float(vertex[0]),
                                       float(vertex[1]),
                                       float(vertex[2])], dtype=np.float32)
                    if in_mm:
                        vertex = vertex / np.float32(10)  # mm → cm
                    vertex = vertex / np.float32(100)  # cm → m
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break

                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True

        vertices = np.matrix(vertices)

        points_path = os.path.join(cfg.LINEMOD_DIR, object_name, 'points.txt')
        if os.path.exists(points_path):
            # 記載されたパスの .txt ファイルが存在する場合
            return vertices
        else:
            np.savetxt(points_path, vertices)
            return vertices


if __name__ == "__main__":
    read_ply = evaluate()
    points = read_ply.read_3d_points_linemod(cfg.linemod_cls_names[0])
