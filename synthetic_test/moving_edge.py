#!/usr/bin/env python3

#./moving_edge
#./moving_edge resolution fps duration output_name

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil

resolution = 64
fps = 30
duration = 5
out_name = 'moving_edge'

if len(sys.argv) == 5:
    resolution = sys.argv[1]
    fps = sys.argv[2]
    duration = sys.argv[3]
    out_name = sys.argv[4]
elif len(sys.argv) != 1:
    exit('Error: illegal parameters')

os.mkdir(out_name)

data = np.zeros((resolution, resolution))
row = np.zeros(resolution)

image_num = fps*duration
move_ratio = int(image_num / resolution)

j = 0
for t in range(0,image_num):
    if t % move_ratio == 0 and int(t/move_ratio) != 0:
        for i in range(0,resolution):
            data[i][j] = 255
        j=(j+1)%resolution
    image = Image.fromarray(np.uint8(data))
    image.save(f'{out_name}/img_{t}.pgm')

subprocess.run(f"ffmpeg -f image2 -framerate 30 -i {out_name}/img_%d.pgm -c:v libx264 -preset veryslow -qp 18 -pix_fmt gray {out_name}.mp4".split())

shutil.rmtree(out_name)
