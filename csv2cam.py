import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image

def read_depth(path):
    with open(path) as file:
        depth_mm = np.loadtxt(os.path.join(path), delimiter=',',skiprows=0)
        img = Image.fromarray(np.rint(depth_mm).astype(np.uint16))
        img_1 = img.resize((1920, 1440))
    return img, img_1

def write_extrinsic(path, dst_dir):
    with open(path) as file:
        i = 0
        extrinsic = np.loadtxt(os.path.join(path), delimiter=',',skiprows=0)
        for line in extrinsic:
            ex_m = np.array([line[3], line[4], line[5], line[0], line[6], line[7], line[8], line[1], line[9], line[10], line[11], line[2], 0.0, 0.0, 0.0, 1.0]).reshape((4,4))
            line = np.linalg.inv(ex_m)
            # cam_content =  str(line[3]) + " " +  str(line[4])  + " " + str(line[5]) + " " + str(line[0]) + " " + str(line[6])+ " " + str(line[7])+ " " + str(line[8])+ " " + str(line[1])+ " " + str(line[9])+ " " + str(line[10])+ " " + str(line[11])+ " " + str(line[2]) + " "
            cam_content =  str(line[0][0]) + " " +  str(line[0][1])  + " " + str(line[0][2]) + " " + str(line[0][3]) + " " + str(line[1][0]) + " " +  str(line[1][1])  + " " + str(line[1][2]) + " " + str(line[1][3]) + " " + str(line[2][0]) + " " +  str(line[2][1])  + " " + str(line[2][2]) + " " + str(line[2][3]) + " "
            f = open(os.path.join(dst_dir, "color", f'{(i+7):06}.cam'), "w")
            f.write(cam_content)
            i += 1
    

def load_depth_image(path):
    depth_dir = os.path.join(path, 'depth')
    depth_org = os.path.join(path, 'depth_org')
    if not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    if not os.path.exists(depth_org):
        os.mkdir(depth_org)
    
    depth_scale = os.path.join(path, 'depth_scale')
    if not os.path.exists(depth_scale):
        os.mkdir(depth_scale)
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    # print("depth_frame",depth_frames)
    depth_frames = [f for f in depth_frames if '.npy' in f or '.csv' in f]
    for i,f in enumerate(depth_frames):
        print(i)
        img, img_rescale = read_depth(f)
        img.save(os.path.join(depth_org , f'{(i):06}.png'))    
        img_rescale.save(os.path.join(depth_scale , f'{(i):06}.png'))   

src_dir = "/Users/minhnd/Documents/test/temp/lidar/sam9/"
dst_dir = "/Users/minhnd/Documents/test/temp/lidar/sam9/"

load_depth_image(src_dir)
# write_extrinsic(src_dir + "/extrinsics.csv", dst_dir)

