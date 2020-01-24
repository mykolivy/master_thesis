#!/usr/bin/env python3

#./moving_edge
#./moving_edge method resolution fps duration output_name
# method can be one of [moving_edge,pixel_random,single_color,checkers]

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import random

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
                if value >= 0:
                    out.write(int(1).to_bytes(1, byteorder='big', signed=False))
                else:
                    out.write(int(2).to_bytes(1, byteorder='big', signed=False))

def add_compact_frame(diff, t, arranged):
    for i, row in enumerate(diff):
        for j, value in enumerate(row):
            if value != 0:
                arranged[i][j].append(t)
                if value >= 0:
                    arranged[i][j].append(1)
                else:
                    arranged[i][j].append(2)

def save_compact_frames(arranged, out):
    for row in arranged:
        for x in row:
            for i in range(0,len(x),2):
                out.write(x[i].to_bytes(4, byteorder='little', signed=False))
                out.write(x[i+1].to_bytes(1, byteorder='big', signed=False))
            out.write(int(0).to_bytes(1, byteorder='big', signed=False))

def moving_edge(resolution, image_num, raw_file, aer_file, caer_file):
    data = np.zeros((resolution, resolution))
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
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
    # Write compact AER data
    save_compact_frames(arranged, caer_file)

def single_color(resolution, image_num, raw_file, aer_file, caer_file):
    data = np.zeros((resolution, resolution))
    zero = np.zeros((resolution, resolution))
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    save_frame(raw_file, data)
    save_frame(aer_file, data)
    save_frame(caer_file, data)
    for t in range(1,image_num):
        save_frame(raw_file, data)
        save_event_frame(zero, t, aer_file)
        add_compact_frame(zero, t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
    # Write compact AER data
    save_compact_frames(arranged, caer_file)

def random_matrix(size):
    return np.random.randint(0, high=2, size=(size, size)) * 255

def pixel_random(resolution, image_num, raw_file, aer_file, caer_file):
    data = random_matrix(resolution) 
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    save_frame(raw_file, data)
    save_frame(aer_file, data)
    save_frame(caer_file, data)
    for t in range(1,image_num):
        prev = data.copy()
        data = random_matrix(resolution)
        save_frame(raw_file, data)
        save_event_frame(np.subtract(data, prev), t, aer_file)
        add_compact_frame(np.subtract(data, prev), t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
    # Write compact AER data
    save_compact_frames(arranged, caer_file)

def checker_matrix(resolution, even=True):
    mat = np.zeros((resolution, resolution))
    if even:
        for row in mat[0::2]:
            row[0::2] = 255
        for row in mat[1::2]:
            row[1::2] = 255
    else:
        for row in mat[0::2]:
            row[1::2] = 255
        for row in mat[1::2]:
            row[0::2] = 255

    return mat

def checkers(resolution, image_num, raw_file, aer_file, caer_file):
    data_even = checker_matrix(resolution, True)
    data_odd = checker_matrix(resolution, False)
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    save_frame(raw_file, data_even)
    save_frame(aer_file, data_even)
    save_frame(caer_file, data_even)
    prev = data_even
    data = data_odd
    for t in range(1,image_num):
        if t%2 == 0:
            data = data_even
            prev = data_odd
        else:
            data = data_odd
            prev = data_even
        save_frame(raw_file, data)
        save_event_frame(np.subtract(data, prev), t, aer_file)
        add_compact_frame(np.subtract(data, prev), t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
            # Write compact AER data
    save_compact_frames(arranged, caer_file)


resolution = 64
fps = 30
duration = 5
np.random.seed(seed=0)
sequence = None
sequences = {'moving_edge': moving_edge, 'pixel_random': pixel_random, 
        'single_color': single_color, 'checkers': checkers}

if len(sys.argv) == 6:
    sequence = sys.argv[1]
    resolution = sys.argv[2]
    fps = sys.argv[3]
    duration = sys.argv[4]
    out_name = sys.argv[5]
elif len(sys.argv) == 2:
    sequence = sys.argv[1]
    if sequence not in sequences:
        exit('Error: unknown sequence name')
else:
    exit('Error: illegal parameters')

out_name = sequence 
os.mkdir(out_name)

raw_file = open(f"{out_name}.raw", "ab+")
aer_file = open(f"{out_name}.aer", "ab+")
caer_file = open(f"{out_name}.caer", "ab+")
create_raw_file(raw_file, resolution, fps, duration)
create_raw_file(aer_file, resolution, fps, duration)
create_raw_file(caer_file, resolution, fps, duration)

image_num = fps*duration

sequences[sequence](resolution, image_num, raw_file, aer_file, caer_file)

# Generate baseline using h.264 codec
subprocess.run(f"ffmpeg -f image2 -framerate 30 -i {out_name}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {out_name}.mp4".split())

shutil.rmtree(out_name)
raw_file.close()
aer_file.close()
caer_file.close()
