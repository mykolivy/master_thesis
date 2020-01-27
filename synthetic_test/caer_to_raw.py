#!/usr/bin/env python3

import numpy as np
import sys
import os
import shutil
import random

header_size = 9

def write_header(resolution, fps, duration, dest):
    dest.write((resolution).to_bytes(4, byteorder='big'))
    dest.write((fps).to_bytes(1, byteorder='big'))
    dest.write((duration).to_bytes(4, byteorder='big'))

def read_start_frame(resolution, source):
    mat = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            mat[i][j] = int.from_bytes(source.read(1), 
                            byteorder='big', 
                            signed=False)
    return mat

def write_frame(data, out):
    for row in np.uint8(data):
        for val in row:
            out.write(int(val).to_bytes(1, byteorder='big'))

def write_repeat_frame(frame, frame_number, dest):
    for i in range(frame_number):
       write_frame(frame, dest) 

def move_to(file, header_size, res, t, i, j):
    file.seek(header_size + t*res*res + i*res + j, 0)

def write_same_val_at(value, start, end, pos, dest):
    for t in range(start, end):
        dest.seek()        

with open(sys.argv[1], 'rb') as source, \
     open(f'{sys.argv[1]}.raw', 'wb') as dest:
    resolution = int.from_bytes(source.read(4), byteorder='big', signed=False)
    fps = int.from_bytes(source.read(1), byteorder='big', signed=False)
    duration = int.from_bytes(source.read(4), byteorder='big', signed=False) 
    prev = read_start_frame(resolution, source)
    frame_number = fps*duration

    print(f"resolution: {resolution}, fps: {fps}, duration: {duration}")
    print(f"start frame: {prev.shape}\n{prev}")
     
    write_header(resolution, fps, duration, dest) 
    write_repeat_frame(prev, frame_number, dest)

    # write event frame contents
    res_sq = resolution*resolution
    first_frame_offset = res_sq
    for i in range(resolution):
        for j in range(resolution):
            prev_t = 0
            t = 0
            pixel_offset = i*resolution + j
            byte = source.read(1)
            while byte != b'\x00':
                # read frame number
                t = int.from_bytes((byte + source.read(3)), 
                        byteorder='little',
                        signed=False)
                frame_offset = t*res_sq

                # read delta value
                delta = int.from_bytes(source.read(2), 
                        byteorder='big', 
                        signed=True)

                # write in-between frame values
                prev_bytes = int(prev[i][j]).to_bytes(1, byteorder='big',
                                                      signed=False)
                for k in range(prev_t+1, t):
                    dest.seek(header_size + k*res_sq + pixel_offset, 0)
                    dest.write(prev_bytes)

                # write frame value
                value = int(prev[i][j] + delta)
                if value > 255 or value < 0:
                    exit(f'Illegal value {value} encountered')
                prev[i][j] = value
                dest.seek(header_size + frame_offset + pixel_offset, 0)
                dest.write(value.to_bytes(1, byteorder='big', signed=False))

                byte = source.read(1)
                prev_t = t

            prev_bytes = int(prev[i][j]).to_bytes(1, byteorder='big',
                                                      signed=False)
            for k in range(t+1, frame_number):
                dest.seek(header_size + k*res_sq + pixel_offset, 0)
                dest.write(prev_bytes)


