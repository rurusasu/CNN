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

        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w, t_p2w, s_p2w

    def pose_p2w(self, RT):
        t, R = RT[:, 3], RT[:, :3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c, self.t_p2w) + self.s_p2w * t
        return np.concatenate([R_w2c, t_w2c[:, None]], 1)
