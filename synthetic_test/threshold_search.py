#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import random
import events

def invert(val):
    if val == 0:
        return 255
    else: 
        return 0

class Frame:
    def __init__(self, resolution):
        self.val = np.zeros((resolution, resolution))
        self.events = {}
        self.res = resolution
    def __init__(self, events, values):
        self.val = values.copy()
        self.events = events
        self.res = values.shape[0]
    def set_from_events(self, events):
        self.events = events.copy()
        for event in events:
            i = int(event / self.res)
            j = event - i*self.res
            self.val[i][j] = invert(self.val[i][j])
    def change_rand(self, rate):
        res_sq = self.res * self.res
        num = int(res_sq * rate)
        available_num = res_sq - len(self.events)
        if available_num <= 0:
            exit("requested rate too big, aborting")
        #population = list(set(range(0, res_sq)).difference(prev))
        population = range(0, res_sq)
        self.set_from_events(random.sample(population, num))

def random_change(resolution, image_num, rate, raw_file, aer_file, caer_file):
    data = np.zeros((resolution, resolution))
    prev = data.copy()
    frame = Frame([], data)
    arranged = [[[] for y in range(0,resolution)] for x in range(0,resolution)]
    # save first frame
    events.save_frame(raw_file, frame.val)
    events.save_frame(aer_file, frame.val)
    events.save_frame(caer_file, frame.val)
    for t in range(1,image_num):
        frame.change_rand(rate) 
        events.save_frame(raw_file, frame.val)
        events.save_event_frame(np.subtract(frame.val, prev), t, aer_file)
        events.add_compact_frame(np.subtract(frame.val, prev), t, arranged)
        Image.fromarray(np.uint8(frame.val)).save(f'{out_name}/{sub_name}/img_{t}.pgm')
        prev = frame.val.copy()
    # Write compact AER data
    events.save_compact_frames(arranged, caer_file)

def log(rate, raw_size, aer_size, caer_size, 
        raw_paq_size, aer_paq_size, caer_paq_size, out):
    out.write(f"RATE: {rate}\n")
    out.write(f"raw size: {raw_size}\naer size: {aer_size}\ncaer size: {caer_size}\n")
    out.write(f"raw.paq size: {paq_raw_size}\naer.paq size: {paq_aer_size}\ncaer.paq size: {paq_caer_size}\n")
    
resolution = 64
fps = 30
duration = 5
np.random.seed(seed=0)

if len(sys.argv) == 5:
    resolution = int(sys.argv[1])
    fps = int(sys.argv[2])
    duration = int(sys.argv[3])
    out_name = sys.argv[4]
elif len(sys.argv) != 1:
    exit('Error: illegal parameters')


precision = 10
paq_raw_size = precision + 1
paq_caer_size = 0 
start = 0
end = 1 
image_num = fps*duration
log_file_name = f'{out_name}.log'
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
log_file = open(log_file_name, "w+")
sub_name = ''
while abs(paq_raw_size - paq_caer_size) > precision:
    rate = start + (end - start) / 2
    print(f"Rate: {rate}")
    log_file.write(f"=============== RATE {rate} ===============")
    sub_name = f"seq_{rate}"

    raw_name = f"{out_name}/{sub_name}.raw" 
    os.makedirs(os.path.dirname(raw_name), exist_ok=True)
    aer_name = f"{out_name}/{sub_name}.aer" 
    os.makedirs(os.path.dirname(aer_name), exist_ok=True)
    caer_name = f"{out_name}/{sub_name}.caer" 
    os.makedirs(os.path.dirname(caer_name), exist_ok=True)
    os.makedirs(os.path.dirname(f"{out_name}/{sub_name}/file.txt"), exist_ok=True)
    with open(raw_name, "ab+") as raw_file,  \
         open(aer_name, "ab+") as aer_file, \
         open(caer_name, "ab+") as caer_file:
        events.create_raw_file(raw_file, resolution, fps, duration)
        events.create_raw_file(aer_file, resolution, fps, duration)
        events.create_raw_file(caer_file, resolution, fps, duration)

        random_change(resolution, image_num, rate, raw_file,aer_file,caer_file)

        # Generate baseline using h.264 codec
        os.system(f"ffmpeg -f image2 -framerate 30 -i {out_name}/{sub_name}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {out_name}/{sub_name}.mp4")

        shutil.rmtree(f'{out_name}/{sub_name}') 

        os.system(f'./lpaq1 0 {raw_name} {raw_name}.paq')
        os.system(f'./lpaq1 0 {aer_name} {aer_name}.paq')
        os.system(f'./lpaq1 0 {caer_name} {caer_name}.paq')
        
        raw_size = os.path.getsize(raw_name)
        aer_size = os.path.getsize(aer_name)
        caer_size = os.path.getsize(caer_name)

        paq_raw_size = os.path.getsize(f'{raw_name}.paq')
        paq_aer_size = os.path.getsize(f'{aer_name}.paq')
        paq_caer_size = os.path.getsize(f'{caer_name}.paq')

        log(rate, raw_size, aer_size, caer_size, 
            paq_raw_size, paq_aer_size, paq_caer_size, log_file)
        diff = abs(paq_raw_size - paq_caer_size)
        log_file.write(f"=============== RATE {rate} | DIFF {diff} ===============")

    if paq_raw_size > paq_caer_size:
        start = start + (end - start) / 2
    else:
        end = start + (end - start) / 2
log_file.close()        
