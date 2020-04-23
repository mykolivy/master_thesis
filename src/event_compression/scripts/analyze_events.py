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

out_str = 'Rate({self.avg_rates}, {self.rate_vars}, {self.min_rates}, {self.max_rates}) | Values({self.avg_vals}, {self.val_vars}, {self.min_vals}, {self.max_vals})'


def extract_frames(video_path, img_folder):
    out = f'{img_folder}/image-%4d.jpg'
    os.system(f'ffmpeg -i {video_path} {out} > /dev/null 2>&1')


def load_frame(frame_path):
    return np.array(Image.open(frame_path))


def log(msg, out):
    print(msg)
    out.write(f'{msg}\n')


class VideoStat:
    avg_rates = 0
    max_rates = 0
    min_rates = math.inf
    rate_vars = 0

    max_vals = 0
    min_vals = math.inf
    val_vars = 0

    pixel_num = 0

    def __init__(self, path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            print("\tExtracting frames...", end='', flush=True)
            extract_frames(video, tmpdir)
            print('\r\tComputing stats...', end='', flush=True)
            frames = list(tmpdir.iterdir())
            self.__process_frames__(frames)
            print('\r\t', end='', flush=True)

    def __process_frames__(self, frames):
        if not frames: raise ValueError()
        frames = sorted(frames)

        rates = []
        vals = []

        prev = load_frame(frames[0])
        shape = prev.shape
        self.pixel_num = shape[0] * shape[1]
        print("\tFirst pass...", end='', flush=True)
        for i in range(1, len(frames)):
            curr = load_frame(frames[i])
            diff = np.subtract(curr, prev)
            self.__update__(diff, rates, vals)
            prev = curr

        rates = [[y / self.pixel_num for y in x] for x in rates]

        print("\r\tComputing average rate...", end='', flush=True)
        self.avg_rates = np.average(rates, axis=(0))
        print("\r\tComputing rate variance...", end='', flush=True)
        self.rate_vars = np.var(rates, axis=(0))
        print("\r\tComputing min and max rate...", end='', flush=True)
        self.max_rates = np.amax(rates, axis=(0))
        self.min_rates = np.amin(rates, axis=(0))
        print("\r\tComputing average values...", end='', flush=True)
        self.avg_vals = np.average(vals, axis=(0))
        print("\r\tComputing value variance...", end='', flush=True)
        self.val_vars = np.var(vals, axis=(0))
        print("\r\tComputing min and max values...", end='', flush=True)
        self.max_vals = np.amax(vals, axis=(0))
        self.min_vals = np.amin(vals, axis=(0))

    def __update__(self, frame_diff, rates, vals):
        abs_diff = np.abs(frame_diff)
        #nonzero = frame_diff[abs_diff != 0]

        #rate = len(nonzero)
        rs = []
        vs = []
        for i in range(3):
            ch = frame_diff[:, :, i]
            rs.append(len(frame_diff[abs_diff[:, :, i] != 0]))
            vs.append(np.average(ch[ch != 0]))
        rates.append(rs)
        vals.append(vs)

    def __str__(self):
        return f'Rate({self.avg_rates}, {self.rate_vars}, {self.min_rates}, {self.max_rates}) | Values({self.avg_vals}, {self.val_vars}, {self.min_vals}, {self.max_vals})'


log_file = open(sys.argv[2], 'w+')
videos = sorted(list(Path(sys.argv[1]).glob('*.mp4')))
video_stats = {}

log(f'Output format: {out_str}', log_file)

for video in videos:
    #try:
    print(f"Analyzing {video}")
    stat = VideoStat(video)
    log(f'{video}: {stat}', log_file)
#except:
#log(f'ERROR: {video} couldn\'t be analysed', log_file)

log_file.close()
