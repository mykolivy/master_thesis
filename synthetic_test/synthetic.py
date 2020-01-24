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
import events

def moving_edge(resolution, image_num, raw_file, aer_file, caer_file):
    data = np.zeros((resolution, resolution))
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    prev = np.zeros((resolution, resolution)) # starting frame
    # save first frame
    events.save_frame(raw_file, prev)
    events.save_frame(aer_file, prev)
    events.save_frame(caer_file, prev)
    row = np.zeros(resolution)
    move_ratio = int(image_num / resolution)
    j = 0
    for t in range(1,image_num):
        if t % move_ratio == 0 and int(t/move_ratio) != 0:
            for i in range(0,resolution):
                data[i][j] = 255
            j=(j+1)%resolution
        events.save_frame(raw_file, data)
        events.save_event_frame(np.subtract(data, prev), t, aer_file)
        events.add_compact_frame(np.subtract(data, prev), t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
        prev = data.copy()
    # Write compact AER data
    events.save_compact_frames(arranged, caer_file)

def single_color(resolution, image_num, raw_file, aer_file, caer_file):
    data = np.zeros((resolution, resolution))
    zero = np.zeros((resolution, resolution))
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    events.save_frame(raw_file, data)
    events.save_frame(aer_file, data)
    events.save_frame(caer_file, data)
    for t in range(1,image_num):
        events.save_frame(raw_file, data)
        events.save_event_frame(zero, t, aer_file)
        events.add_compact_frame(zero, t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
    # Write compact AER data
    events.save_compact_frames(arranged, caer_file)

def random_matrix(size):
    return np.random.randint(0, high=2, size=(size, size)) * 255

def pixel_random(resolution, image_num, raw_file, aer_file, caer_file):
    data = random_matrix(resolution) 
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    events.save_frame(raw_file, data)
    events.save_frame(aer_file, data)
    events.save_frame(caer_file, data)
    for t in range(1,image_num):
        prev = data.copy()
        data = random_matrix(resolution)
        events.save_frame(raw_file, data)
        events.save_event_frame(np.subtract(data, prev), t, aer_file)
        events.add_compact_frame(np.subtract(data, prev), t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
    # Write compact AER data
    events.save_compact_frames(arranged, caer_file)

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
    events.save_frame(raw_file, data_even)
    events.save_frame(aer_file, data_even)
    events.save_frame(caer_file, data_even)
    prev = data_even
    data = data_odd
    for t in range(1,image_num):
        if t%2 == 0:
            data = data_even
            prev = data_odd
        else:
            data = data_odd
            prev = data_even
        events.save_frame(raw_file, data)
        events.save_event_frame(np.subtract(data, prev), t, aer_file)
        events.add_compact_frame(np.subtract(data, prev), t, arranged)
        Image.fromarray(np.uint8(data)).save(f'{out_name}/img_{t}.pgm')
            # Write compact AER data
    events.save_compact_frames(arranged, caer_file)


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
events.create_raw_file(raw_file, resolution, fps, duration)
events.create_raw_file(aer_file, resolution, fps, duration)
events.create_raw_file(caer_file, resolution, fps, duration)

image_num = fps*duration

sequences[sequence](resolution, image_num, raw_file, aer_file, caer_file)

# Generate baseline using h.264 codec
subprocess.run(f"ffmpeg -f image2 -framerate 30 -i {out_name}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {out_name}.mp4".split())

shutil.rmtree(out_name)
raw_file.close()
aer_file.close()
caer_file.close()
