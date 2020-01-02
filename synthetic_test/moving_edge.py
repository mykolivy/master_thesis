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

def create_raw_file(f, resolution, fps, duration):
    f.write((resolution).to_bytes(4, byteorder='big'))
    f.write((fps).to_bytes(1, byteorder='big'))
    f.write((duration).to_bytes(4, byteorder='big'))

def save_frame(out, data):
    for row in np.uint8(data):
        for j in row:
            out.write(int(j).to_bytes(1, byteorder='big'))
        
def save_event_frame(diff, t, out):
    for i, row in enumerate(diff):
        for j, value in enumerate(row):
            if value != 0:
                out.write(i.to_bytes(4, byteorder='big', signed=False))
                out.write(j.to_bytes(4, byteorder='big', signed=False))
                out.write(t.to_bytes(4, byteorder='little', signed=False))
                out.write((int(value)).to_bytes(2, byteorder='big', signed=True))

def add_compact_frame(diff, t, arranged):
    for i, row in enumerate(diff):
        for j, value in enumerate(row):
            if value != 0:
                arranged[i][j].append(t) 
                arranged[i][j].append(int(value))

def save_compact_frames(arranged, out):
    for row in arranged:
        for x in row:
            for i in range(0,len(x),2):
                out.write(x[i].to_bytes(4, byteorder='little', signed=False))
                out.write(x[i+1].to_bytes(2, byteorder='big', signed=True))
            out.write(int(0).to_bytes(1, byteorder='big', signed=False))

if len(sys.argv) == 5:
    resolution = sys.argv[1]
    fps = sys.argv[2]
    duration = sys.argv[3]
    out_name = sys.argv[4]
elif len(sys.argv) != 1:
    exit('Error: illegal parameters')

os.mkdir(out_name)
if os.path.isfile(f"{out_name}.raw"):
    exit('File already exists')

raw_file = open(f"{out_name}.raw", "ab+")
aer_file = open(f"{out_name}.aer", "ab+")
caer_file = open(f"{out_name}.caer", "ab+")
create_raw_file(raw_file, resolution, fps, duration)
create_raw_file(aer_file, resolution, fps, duration)
create_raw_file(caer_file, resolution, fps, duration)

data = np.zeros((resolution, resolution))
arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
image_num = fps*duration

# Moving edge
prev = np.zeros((resolution, resolution)) # starting frame
# save first frame
save_frame(raw_file, prev)
save_frame(aer_file, prev)
save_frame(caer_file, prev)
row = np.zeros(resolution)
move_ratio = int(image_num / resolution)
j = 0
for t in range(1,image_num):
    if t % move_ratio == 0 and int(t/move_ratio) != 0:
        for i in range(0,resolution):
            data[i][j] = 255
        j=(j+1)%resolution
    
    save_frame(raw_file, data)
    save_event_frame(np.subtract(data, prev), t, aer_file)
    add_compact_frame(np.subtract(data, prev), t, arranged)
    Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')

    prev = data.copy()

# Generate baseline using h.264 codec
subprocess.run(f"ffmpeg -f image2 -framerate 30 -i {out_name}/img_%d.pgm -c:v libx264 -preset veryslow -qp 18 -pix_fmt gray {out_name}.mp4".split())

# Write compact AER data
save_compact_frames(arranged, caer_file)

shutil.rmtree(out_name)
raw_file.close()
aer_file.close()
caer_file.close()
