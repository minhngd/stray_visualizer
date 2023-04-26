import numpy as np
import open3d as o3d
import os

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def read_data(path):
    
    depth_dir = os.path.join(path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]

    rgb_dir = os.path.join(path, 'color')
    rgb_frames = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
    rgb_frames = [f for f in rgb_frames if '.npy' in f or '.jpeg' in f]

    odometry_log_path = os.path.join(path,  "odometry.log")
    print("len: ", len(depth_frames), len(rgb_frames))
    
    return {  'depth_frames': depth_frames, 'rgb_frames': rgb_frames, "odometry_log_path": odometry_log_path }

redwood_rgbd = read_data("/Users/hoangle/Documents/test/temp/lidarexport/sam9")
camera_poses = read_trajectory(redwood_rgbd["odometry_log_path"])

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(redwood_rgbd["rgb_frames"][i])
    depth = o3d.io.read_image(redwood_rgbd["depth_frames"][i])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))

print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
                                  front=[0.5297, -0.1873, -0.8272],
                                  lookat=[2.0712, 2.0312, 1.7251],
                                  up=[-0.0558, -0.9809, 0.1864],
                                  zoom=0.47)