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
import argparse

# Define script interface
parser = argparse.ArgumentParser(description='Perform binary search of event \
        rate performance threshold')
parser.add_argument('sequence', help='type of sequences generated', 
                    choices=['moving_edge', 'random_pixel', 'single_color', 
                             'checkers', 'rate_random_flip'])
parser.add_argument('-r', '--res', dest='res', action='store', default=[64, 64],
                    nargs=2, type=int,
                    help='Resolution of the generated sequences: y x')
parser.add_argument('-f', '--fps', dest='fps', action='store', default=30,
                    help='Framerate of the generated sequences')
parser.add_argument('-d', '--duration', dest='duration', action='store',
                    type=int, default=5,
                    help='Duration of the generated sequences')
parser.add_argument('--rate', type=float, default=0.5)
parser.add_argument('--value', type=int, default=0)
parser.add_argument('--range', type=int, default=[0, 255], nargs=2)
parser.add_argument('out', help='output file')
args = parser.parse_args()
print(args)

sequence_config = seq.SequenceConfig(args.res, args.fps, args.duration)
frame_iterators = {
        'moving_edge': seq.MovingEdgeFrameIterator(sequence_config), 
        'random_pixel': seq.RandomPixelFrameIterator(
                        (args.range[0],args.range[1]), sequence_config),
        'single_color': seq.SingleColorFrameIterator(args.value, sequence_config), 
        'checkers': seq.CheckersFrameIterator(sequence_config),
        'rate_random_flip': seq.RandomBinaryChangeFrameIterator(
                            args.rate, sequence_config)
}
format_iterators = {
        'aer': events.AERIterator,
        'caer': events.CAERIterator
}

out_name = '.'.join(args.out.split('.')[:-1]) 
format = args.out.split('.')[-1]
frame_iterator = frame_iterators[args.sequence]
file_name = f"{out_name}.{format}"

os.makedirs(os.path.dirname(file_name), exist_ok=True)
out = open(file_name, "wb+")
events.create_raw_file(out, args.res, args.fps, args.duration)
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
