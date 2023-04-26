import os
import open3d as o3d
import glob
import numpy as np
import trimesh

folder_path = "/Users/minhnd/Downloads/sam2_optimization/"
# instriction_path = folder_path + "camera_intrinsic.json"
odometry_log_path = folder_path + "odometry_optimaze.log"
depth_paths = glob.glob(folder_path + "depth_scale/*.png")
depth_paths.sort()
confidence_paths = glob.glob(folder_path + "confidence/*.png")
confidence_paths.sort()
color_paths = glob.glob(folder_path + "color/*.jpeg")
color_paths.sort()
reconstruction_path = folder_path + "scan.obj"

def load_fountain_dataset():
    rgbd_images = []
    fountain_rgbd_dataset = o3d.data.SampleFountainRGBDImages()
    for i in range(len(depth_paths)):
        depth = o3d.io.read_image(depth_paths[i])
        color = o3d.io.read_image(color_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        odometry_log_path)
    
    mesh = o3d.io.read_triangle_mesh(reconstruction_path)

    # pcd = o3d.io.read_point_cloud(reconstruction_path)
    # pcd.estimate_normals()
    
    # # estimate radius for rolling ball
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    # # print('run Ball Pivoting surface reconstruction')
    # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    # #         pcd,
    # #         o3d.utility.DoubleVector([radius, radius * 2]))

    # print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=8)
    #     vertices_to_remove = densities < np.quantile(densities, 0.1)
    #     mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh, rgbd_images, camera_trajectory

# Load dataset
mesh, rgbd_images, camera_trajectory = load_fountain_dataset()

# Optimize texture and save the mesh as texture_mapped.ply
# This is implementation of following paper
# Q.-Y. Zhou and V. Koltun,
# Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
# SIGGRAPH 2014

# Run rigid optimization.
# mesh_obj = o3d.io.read_triangle_mesh("/Volumes/Data/Workspace/VMODEV/MBI/StrayVisualizer-main/output.obj")
maximum_iteration = 100
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(
            maximum_iteration=maximum_iteration))

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.5399,
                                  front=[0.0665, -0.1107, -0.9916],
                                  lookat=[0.7353, 0.6537, 1.0521],
                                  up=[0.0136, -0.9936, 0.1118])