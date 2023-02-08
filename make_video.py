import argparse
import os
import numpy as np
from PIL import Image
import skvideo
from matplotlib import cm
from skvideo import io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    return parser.parse_args()

def main():
    flags = parse_args()
    flags = "/Users/hoangle/Documents/test/temp/6eb0146939"

    #images = sorted(os.listdir(os.path.join(flags.data, 'depth')))
    images = sorted(os.listdir(os.path.join(flags, 'depth')))
    
    frames = []

    for i, image in enumerate(images):
        # pprint(f"Reading image {image}", end='\r')
        
        # path = os.path.join(flags.data, 'depth', image)
        path = os.path.join(flags, 'depth', image)
        
        # pprint(path)
        # depth = np.load(path, allow_pickle=True)

        image = Image.open(path)
        # convert image to numpy array
        depth = np.asarray(image)

        # print(depth.shape)

        max_depth = 7.5
        depth_m = depth / 1000.0
        depth_map = np.clip(1.0 - depth_m / max_depth, 0.0, 1.0)
        depth_map = cm.inferno(depth_map)

        frames.append((depth_map * 255).astype(np.uint8))

    # print("Frame size =", len(frames))
    # writer = io.FFmpegWriter(os.path.join(flags.data, 'depth_video.mp4'))
    writer = io.FFmpegWriter(os.path.join(flags, 'depth_video.mp4'), outputdict={ "-vcodec" : "libx264", "-pix_fmt" : "yuv420p"})

    try:
        for i, frame in enumerate(frames):
            # print(f"Writing frame {i:06}" + " " * 10, end='\r')
            writer.writeFrame(frame)
    finally:
        writer.close()

if __name__ == '__main__':
    main()

# python StrayVisualizer/make_video.py ./6eb0146939