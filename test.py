from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import open3d as o3d

def format_data(old_fold, new_fold):
    cam = glob.glob(old_fold + "*.cam")
    cam.sort()
    return cam

def load_depth(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        print("hehehh")
        depth_mm = np.array(Image.open(path))
    # depth_scale = depth_mm/1000
    # img = Image.fromarray(depth_scale)
    # img.save("abc.png")
    print(depth_mm.shape, np.amax(depth_mm), "\n", depth_mm)
    # print(depth_mm)
    # for i in depth_mm[0]:
    #     print(i)
        # if i > 10: 
        #     break
    # depth_m = depth_mm.astype(np.float32) / 1000.0
    
    # im = Image.fromarray(depth_m)
    # im.save("hehe.png")

def load_npz(path):
    # f = np.load(path)
    f = np.array(Image.open(path))
    print(f[0][0][0], f[0][0][1], f[0][0][2], f[0][0][3])
    # for i in range(10):
    #     for j in range(10):
            
    #         print(cmyk_to_luminance(f[i][j][3], f[i][j][2], f[i][j][1], f[i][j][0]))
    # print(f["depth"].shape)
    # print(cmyk_to_luminance(f["depth"][0]))
    # print(f["w2c"])
    # print(f.files)

def print_npz_depth(path):
    f = np.load(path)
    print(f['depth'].shape)

def cmyk_to_luminance(c, m, y, k):
    c = 1 - ( c / 255 )
    m = 1 - ( m / 255 )
    y = 1 - ( y / 255 )
    k = 1 - ( k /255 )  
    c = c * (1 - k) + k
    m = m * (1 - k) + k
    y = y * (1 - k) + k

    r, g, b = (1 - c), (1 - m), (1 - y)
    print(r,g,b)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

fountain_rgbd_dataset = o3d.data.SampleFountainRGBDImages()

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

    for i, (T_WC) in enumerate(zip(data['poses'])):
        print(f"Integrating frame {i:06}", end='\r')
        depth_path = data['depth_frames'][i]
        rgb_path = data['rgb_frames'][i]
        
        depth = load_depth(depth_path)
        
        rgb = load_rgb(rgb_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)

        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC[0]))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh
