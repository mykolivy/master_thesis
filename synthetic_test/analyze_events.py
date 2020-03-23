#!/usr/bin/env python3

# Given a folder full of videos outputs for each of them and an average values:
#   - average rate of pixel changes
#   - highest rate of pixel changes
#   - lowest rate of pixel changes
#   - variance of rate of pixel change
#   - average intensity value change
#   - highest intensity value change
#   - lowest intensity value change
#   - variance of rate of pixel change

from PIL import Image
from pathlib import Path
import numpy as np
import sys
import os
import subprocess
import shutil
import random
import tempfile
import math

def extract_frames(video_path, img_folder):
    out = f'{img_folder}/image-%4d.jpg'
    os.system(f'ffmpeg -i {video_path} {out} > /dev/null 2>&1')

def load_frame(frame_path):
    return np.array(Image.open(frame_path))

def log(msg, out):
    print(msg)
    out.write(f'{msg}\n')

class VideoStat:
    avg_rate = 0
    max_rate = 0
    min_rate = math.inf
    rate_var = 0

    max_val = 0
    min_val = math.inf
    val_var = 0

    pixel_num = 0

    def __init__(self, path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            print("\tExtracting frames...", end='', flush=True)
            extract_frames(video, tmpdir)
            print('\r\tExtracted frames succefully')
            frames = list(tmpdir.iterdir())
            self.__process_frames__(frames)
     
    def __process_frames__(self, frames):
        if not frames: raise ValueError()
        frames = sorted(frames)
        
        rates = []
        vals = []

        prev = load_frame(frames[0])
        self.pixel_num = prev.size
        for i in range(1, len(frames)):
            curr = load_frame(frames[i])
            diff = curr - prev
            self.__update__(diff, rates, vals)
            prev = curr.copy()

        rates = [x / self.pixel_num for x in rates]
        self.avg_rate = np.average(rates)
        self.rate_var = np.var(rates) 
        self.max_rate = np.amax(rates)
        self.min_rate = np.amin(rates)
        self.val_var = np.var(vals)

    def __update__(self, frame_diff, rates, vals):
        nonzero = frame_diff[frame_diff != 0]
       
        rate = len(nonzero)
        rates.append(rate)
        
        avg_val = np.average(nonzero)
        vals.append(avg_val)
        
        min_val = np.amin(nonzero)
        max_val = np.amax(nonzero)
        if self.min_val > min_val: self.min_val = min_val
        if self.max_val < max_val: self.max_val = max_val
            
    def __str__(self):
        return f'Rate({self.avg_rate}, {self.rate_var}, {self.min_rate}, {self.max_rate}) | Values({self.val_var}, {self.min_val}, {self.max_val})'

log_file = open(sys.argv[2], 'w+')
videos = Path(sys.argv[1]).glob('*.mp4')
video_stats = {}

for video in videos:
    try: 
        print(f"Analyzing {video}")
        stat = VideoStat(video)
        log(f'{video}: {stat}', log_file)
    except:
        log(f'ERROR: {video} couldn\'t be analysed', log_file)
            
log_file.close()
