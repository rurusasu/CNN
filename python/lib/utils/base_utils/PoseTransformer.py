class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    blender_models = {}

    def __init__(self, object_name):
        self.object_name = object_name
        self.blender_model_path = os.path.join(cfg.PVNET_LINEMOD_DIR, '{}/{}.ply'.format(object_name, object_name))
        self.orig_model_path = os.path.join(cfg.LINEMOD_ORIG, '{}/mesh.ply'.format(class_type))
        self.xyz_pattern = os.path.join(cfg.OCCLUTION_LINEMOD, 'models/{}/{}.xyz')
        self.model_aligner = ModelAligner(object_name)

    def orig_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.gettranslation_transform())
