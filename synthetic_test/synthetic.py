#!/usr/bin/env python3

#./moving_edge method output
#./moving_edge method resolution fps duration output
# method can be one of [moving_edge,pixel_random,single_color,checkers]

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import random
import events
import sequence as seq

resolution = 64
fps = 30
duration = 5
np.random.seed(seed=0)
out_name = 'default_name'
method = sys.argv[1]
format = sys.argv[-1].split('.')[-1]
out_name = '.'.join(sys.argv[-1].split('.')[:-1]) 
if len(sys.argv) == 6:
    resolution = int(sys.argv[2])
    fps = int(sys.argv[3])
    duration = int(sys.argv[4])

sequence_config = seq.SequenceConfig(resolution, fps, duration)
frame_iterators = {
        'moving_edge': seq.MovingEdgeFrameIterator(sequence_config), 
        'random_pixel': seq.RandomPixelFrameIterator((0,2), sequence_config),
        'single_color': seq.SingleColorFrameIterator(255, sequence_config), 
        'checkers': seq.CheckersFrameIterator(sequence_config)
}
format_iterators = {
        'aer': events.AERIterator,
        'caer': events.CAERIterator
}

frame_iterator = frame_iterators[method]
file_name = f"{out_name}.{format}"

os.makedirs(os.path.dirname(file_name), exist_ok=True)
out = open(file_name, "wb+")
events.create_raw_file(out, resolution, fps, duration)
events.save_frame(out, frame_iterator.start_frame)
if format == 'raw':
    os.mkdir(out_name)
    for i, frame in enumerate(frame_iterator):
        events.save_frame(out, frame)
        Image.fromarray(np.uint8(frame)).save(f'{out_name}/img_{i}.pgm')
    subprocess.run(f"ffmpeg -f image2 -framerate 30 -i {out_name}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {out_name}.mp4".split())
    shutil.rmtree(out_name)
else:
    format_iterator = format_iterators[format]
    for chunk in next(format_iterator(frame_iterator)):
        out.write(chunk)
