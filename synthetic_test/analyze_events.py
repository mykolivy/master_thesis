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

input_folder = Path(sys.argv[1])
all_files = input_folder.glob('**/*')
grand_data = [[], [], []]
grand_rate = [[], [], []]
log_file = open(sys.argv[2], 'w+')
for file_name in [x for x in all_files if x.is_file()]:
    #local_data = [[], [], []]
    #local_rate = [[], [], []]
    img_folder = 'temp_img_folder'
    os.mkdir(img_folder)
    os.system(f'ffmpeg -i {file_name} {img_folder}/image-%4d.png > /dev/null')
    prev = None 
    first = True
    diffs = []
    rates = []
    for img_name in [x for x in sorted(Path(img_folder).glob('**/*')) if x.is_file()]:
        img = Image.open(img_name)
        values = np.array(img)
        pixel_num = values.shape[0] * values.shape[1]
        if first:
            print(f'{file_name} {values.shape}')
            log_file.write(f'{file_name} {values.shape}')
        else:
            diff = np.subtract(values, prev)
            diffs.append(diff)
            rates.append(np.count_nonzero(diff, axis=(0,1)) / pixel_num)
            #event_num = [0,0,0]
            #for row in diff:
                #for channels in row:
                    #for i, x in enumerate(channels):
                        #if x != 0:
                            #local_data[i].append(x)
                            #event_num[i] += 1
            #for i in range(3):
                #local_rate[i].append(event_num[i] / pixel_num)

        prev = values
        first = False
    print(f'Mean rate: {np.mean(rates, axis=0)}\n')
    log_file.write(f'Mean rate: {np.mean(rates, axis=0)}\n')

    shutil.rmtree(img_folder) 
log_file.close()
