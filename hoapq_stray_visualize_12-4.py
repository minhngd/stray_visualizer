import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from PIL import Image
import skvideo.io
# import simd
# from scipy.spatial.transform import rotation
import csv
file  = open('odometry_optimized.csv', mode='w')
   

description = """
This script visualizes datasets collected using the Stray Scanner app.
"""

usage = """
Basic usage: python stray_visualize.py <path-to-dataset-folder>
"""

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
MAX_DEPTH = 5.0

# Pose graph optimization
voxel_size = 0.02
pcds_down = []
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument('--trajectory', '-t', action='store_true', help="Visualize the trajectory of the camera as a line.")
    parser.add_argument('--frames', '-f', action='store_true', help="Visualize camera coordinate frames from the odometry file.")
    parser.add_argument('--point-clouds', '-p', action='store_true', help="Show concatenated point clouds.")
    parser.add_argument('--integrate', '-i', action='store_true', help="Integrate point clouds using the Open3D RGB-D integration pipeline, and visualize it.")
    parser.add_argument('--mesh-filename', type=str, help='Mesh generated from point cloud integration will be stored in this file. open3d.io.write_triangle_mesh will be used.', default=None)
    parser.add_argument('--every', type=int, default=1, help="Show only every nth point cloud and coordinate frames. Only used for point cloud and odometry visualization.")
    parser.add_argument('--voxel-size', type=float, default=0.5, help="Voxel size in meters to use in RGB-D integration.")
    parser.add_argument('--confidence', '-c', type=int, default=1,
            help="Keep only depth estimates with confidence equal or higher to the given value. There are three different levels: 0, 1 and 2. Higher is more confident.")
    return parser.parse_args()

def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

def read_data(flags):
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    extrinsiccsv = np.loadtxt(os.path.join(flags.path, 'extrinsics.csv'), delimiter=',',skiprows=0)
    poses = []
    tem_poses = []
    extrinsics = []
    
    for line in extrinsiccsv:
        # ex_m = np.array([line[3], line[6], line[9], line[0], line[4], line[7], line[10], line[1], line[5], line[8], line[11], line[2], 0.0, 0.0, 0.0, 1.0]).reshape((4,4))
        ex_m = np.array([line[3], line[4], line[5], line[0], line[6], line[7], line[8], line[1], line[9], line[10], line[11], line[2], 0.0, 0.0, 0.0, 1.0]).reshape((4,4))
        extrinsics.append(ex_m)

    depth_dir = os.path.join(flags.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    # depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    depth_frames = [f for f in depth_frames if '.csv' in f or '.png' in f]

    rgb_dir = os.path.join(flags.path, 'color')
    rgb_frames = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
    rgb_frames = [f for f in rgb_frames if '.npy' in f or '.jpg' in f]
    print("len: ", len(poses), len(depth_frames), len(rgb_frames), len(extrinsics), len(intrinsics))
    return { 'poses': poses, 'depth_frames': depth_frames, 'rgb_frames': rgb_frames, 'extrinsics': extrinsics, 'intrinsics': intrinsics }

def load_depth(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32)/1000
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    print("filter: ", filter_level, "\n", depth_m.shape)
    return o3d.geometry.Image(depth_m)

def load_csv_depth(path, confidence=None, filter_level=0):  
    depth_mm = np.loadtxt(os.path.join(path), delimiter=',',skiprows=0)
    depth_m = depth_mm.astype(np.float32)/1000
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    print("filter: ", filter_level, "\n", depth_m.shape)
    return o3d.geometry.Image(depth_m)
    

def load_rgb(path):
    rgb_mm = np.array(Image.open(path))
    rgb = Image.fromarray(rgb_mm)
    rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
    # rgb_m = rgb_mm.resize(DEPTH_WIDTH, DEPTH_HEIGHT)
    print("rgb_mm: ", type(rgb_mm), rgb_mm.shape)
    # rgb_m = rgb_mm.astype(np.float32) / 1000.0
    rgb = np.array(rgb)
    return rgb

def load_confidence(path):
    return np.array(Image.open(path))

def get_intrinsics(intrinsics):
    """
    Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
    """
    intrinsics_scaled = _resize_camera_matrix(intrinsics, DEPTH_WIDTH / 640, DEPTH_HEIGHT / 480)
    return o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], cx=intrinsics_scaled[0, 2], cy=intrinsics_scaled[1, 2])

def trajectory(flags, data):
    """
    Returns a set of lines connecting each camera poses world frame position.
    returns: [open3d.geometry.LineSet]
    """
    line_sets = []
    previous_pose = None
    for i, T_WC in enumerate(data['poses']):
        if previous_pose is not None:
            points = o3d.utility.Vector3dVector([previous_pose[:3, 3], T_WC[:3, 3]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets.append(line)
        previous_pose = T_WC
    return line_sets

def show_frames(flags, data):
    """
    Returns a list of meshes of coordinate axes that have been transformed to represent the camera matrix
    at each --every:th frame.

    flags: Command line arguments
    data: dict with keys ['poses', 'intrinsics']
    returns: [open3d.geometry.TriangleMesh]
    """
    frames = [o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.25, np.zeros(3))]
    for i, T_WC in enumerate(data['poses']):
        if not i % flags.every == 0:
            continue
        # print(f"Frame {i}", end="\r")
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        frames.append(mesh.transform(T_WC))
    return frames

def point_clouds(flags, data):
    """
    Converts depth maps to point clouds and merges them all into one global point cloud.
    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    returns: [open3d.geometry.PointCloud]
    """
    pcs = []
    intrinsics = get_intrinsics(data['intrinsics'])
    pc = o3d.geometry.PointCloud()
    meshes = []
    for i, (T_WC) in enumerate(zip(data['extrinsics'])):
        if i < 10000:
            T_CW = np.linalg.inv(T_WC[0])
            if i % flags.every != 0:
                continue
            
            # confidence = load_confidence(os.path.join(flags.path, 'confidence', f'conf_{(i ):06}.png'))
            depth_path = data['depth_frames'][i]
            rgb_path = data['rgb_frames'][i]
            print("Flags: ", flags)
            depth = load_depth(depth_path, filter_level=flags.confidence)
            # depth = load_csv_depth(depth_path, confidence, filter_level=flags.confidence)
            rgb = load_rgb(rgb_path)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), depth,
                depth_scale=1, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_CW)
            pcs.append(pc)
    return pcs

# def integrate(flags, data):
#     """
#     Integrates collected RGB-D maps using the Open3D integration pipeline.

#     flags: command line arguments
#     data: dict with keys ['intrinsics', 'poses']
#     Returns: open3d.geometry.TriangleMesh
#     """
#     volume = o3d.pipelines.integration.ScalableTSDFVolume(
#             voxel_length=flags.voxel_size,
#             sdf_trunc=0.05,
#             color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

#     # intrinsics = get_intrinsics(data['intrinsics'])
#     for i, (T_WC) in enumerate(zip(data['poses'])):
#         print(f"Integrating frame {i:06}", end='\r')
#         depth_path = data['depth_frames'][i]
#         rgb_path = data['rgb_frames'][i]
#         depth = load_depth(depth_path)
#         # rgb = Image.fromarray(rgb)
#         # rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
#         # rgb = np.array(rgb)
#         # rgb = load_rgb(rgb_path)
#         # depth = load_csv_depth(depth_path)
#         rgb = load_rgb(rgb_path)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d.geometry.Image(rgb), depth,
#             depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)

#         volume.integrate(rgbd, get_intrinsics(data['intrinsics'][i]), np.linalg.inv(T_WC[0]))
#     mesh = volume.extract_triangle_mesh()
#     mesh.compute_vertex_normals()
#     return mesh
def integrate(flags, data):
    """
    Integrates collected RGB-D maps using the Open3D integration pipeline.

    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    Returns: open3d.geometry.TriangleMesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=flags.voxel_size,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsics = get_intrinsics(data['intrinsics'])

    # rgb_path = os.path.join(flags.path, 'rgb.mp4')
    # video = skvideo.io.vreader(rgb_path)
    for i, (T_WC) in enumerate(zip(data['poses'])):
        print(f"Integrating frame {i:06}", end='\r')
        # depth_path = data['depth_frames'][i]
        # depth = load_depth(depth_path)
        # rgb = Image.fromarray(rgb)
        # rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        # rgb = np.array(rgb)
        depth_path = data['depth_frames'][i]
        rgb_path = data['rgb_frames'][i]
        print("Flags: ", flags)
        depth = load_depth(depth_path)
        # depth = load_csv_depth(depth_path, confidence, filter_level=flags.confidence)
        rgb = load_rgb(rgb_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)

        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC[0]))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def validate(flags):
    # if not os.path.exists(os.path.join(flags.path, 'rgb.mp4')):
    #     absolute_path = os.path.abspath(flags.path)
    #     print(f"The directory {absolute_path} does not appear to be a directory created by the Stray Scanner app.")
    #     return False
    return True

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    print(n_pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

# def main():
#     flags = read_args()

#     if not validate(flags):
#         return

#     if not flags.frames and not flags.point_clouds and not flags.integrate:
#         flags.frames = True
#         flags.point_clouds = True
#         flags.trajectory = True

#     data = read_data(flags)
#     geometries = []
#     if flags.trajectory:
#         geometries += trajectory(flags, data)
#     if flags.frames:
#         geometries += show_frames(flags, data)
#     if flags.point_clouds:
#         pcds = point_clouds(flags, data)
#         for pcd in pcds:
#             pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
#             pcd_down.estimate_normals()
#             pcds_down.append(pcd_down)

#         print("Full registration ...")
#         with o3d.utility.VerbosityContextManager(
#                 o3d.utility.VerbosityLevel.Debug) as cm:
#             pose_graph = full_registration(pcds_down,
#                                         max_correspondence_distance_coarse,
#                                         max_correspondence_distance_fine)
            
#         print("Optimizing PoseGraph ...")
#         option = o3d.pipelines.registration.GlobalOptimizationOption(
#             max_correspondence_distance=max_correspondence_distance_fine,
#             edge_prune_threshold=0.25,
#             reference_node=0)
#         with o3d.utility.VerbosityContextManager(
#                 o3d.utility.VerbosityLevel.Debug) as cm:
#             o3d.pipelines.registration.global_optimization(
#                 pose_graph,
#                 o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#                 o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
#                 option)
            
#         print("Transform points and display : ", len(pcds_down))
#         for point_id in range(len(pcds_down)):
#             print(pose_graph.nodes[point_id].pose)
#             pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
#             print("Pose new", pose_graph.nodes[point_id].pose)
#             extrinsic = pose_graph.nodes[point_id].pose
#             current_transform = np.linalg.inv(extrinsic)
            
#             trans = current_transform
#             roat = R.from_matrix([
#                             [trans[0][0], trans[0][1], trans[0][2]],
#                             [trans[1][0], trans[1][1], trans[1][2]],
#                             [trans[2][0], trans[2][1], trans[2][2]]]).as_quat()
#             qx = roat[0]
#             qy = roat[1]
#             qz = roat[2]
#             qw = roat[3]
#             x = trans[0][3]
#             y = trans[1][3]
#             z = trans[2][3]
#             # print("posdfsfsfs: ", )
#             odometry =   str(x) + ", " + str(y) + ", " + str(z) + ", " + str(qx) + ", " + str(qy) + ", " + str(qz) + ", "  + str(qw) + "\n" 
#             file.write(odometry)
#         geometries += pcds_down
#     if flags.integrate:
#         mesh = integrate(flags, data)
#         if flags.mesh_filename is not None:
#             o3d.io.write_triangle_mesh(flags.mesh_filename, mesh)
#         geometries += [mesh]
#     o3d.visualization.draw_geometries(geometries)

def main():
    flags = read_args()

    if not validate(flags):
        return

    if not flags.frames and not flags.point_clouds and not flags.integrate:
        flags.frames = True
        flags.point_clouds = True
        flags.trajectory = True

    data = read_data(flags)
    geometries = []
    if flags.trajectory:
        geometries += trajectory(flags, data)
    if flags.frames:
        geometries += show_frames(flags, data)
    if flags.point_clouds:
        geometries += point_clouds(flags, data)
    if flags.integrate:
        mesh = integrate(flags, data)
        if flags.mesh_filename is not None:
            o3d.io.write_triangle_mesh(flags.mesh_filename, mesh)
        geometries += [mesh]
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()

