SMPL_GFEODIST = 'data/essentials/geodesics/smpl/smpl_neutral_geodesic_dist.npy'
SMPL_MODEL_DIR = 'data/models/'
PRIOR_PATH = ''

JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

reproj_joints = [
    'OP Nose', 'OP LEar', 'OP REar', 'OP LShoulder', 'OP RShoulder',
    'OP LElbow', 'OP RElbow', 'OP LWrist', 'OP RWrist',
    'OP LHip', 'OP RHip', 'OP LKnee', 'OP RKnee',
    'OP LAnkle', 'OP RAnkle'
]

reproj_idx = [JOINT_MAP[j] for j in reproj_joints]

smooth_idx = reproj_idx.copy()
smooth_idx.append(6) # for center of torso_upperarm