#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import random
import event_compression.codec.aer as aer
import event_compression.codec.caer as caer
from . import sequence as seq
import argparse
import event_compression.codec.util.io as io
import tempfile
from pathlib import Path
from event_compression.codec import codecs
from . import sequences

formats = codecs()
formats['raw'] = None # Register additional format
sequences = sequences()

# Define script interface
parser = argparse.ArgumentParser(description='Perform binary search of event \
        rate performance threshold')
parser.add_argument('format', help='format of the output file',
                    choices=list(formats.keys()))
parser.add_argument('sequence', help='type of sequences generated', 
                    choices=list(sequences.keys()))
parser.add_argument('-r', '--res', dest='res', action='store', default=[64, 64],
                    nargs=2, type=int,
                    help='Resolution of the generated sequences: y x')
parser.add_argument('--fps', dest='fps', action='store', default=30,
                    help='Framerate of the generated sequences', type=int)
parser.add_argument('-d', '--duration', dest='duration', action='store',
                    type=int, default=5,
                    help='Duration of the generated sequences')
parser.add_argument('--rate', type=float, default=0.5)
parser.add_argument('--value', type=int, default=0)
parser.add_argument('--range', type=int, default=[0, 255], nargs=2)
parser.add_argument('out', help='output file')
args = parser.parse_args()

print(args)

sequence_config = seq.Config(args.res, args.fps, args.duration, value=args.value, rate=args.rate, val_range=args.range)
frame_iterator = sequences[args.sequence](sequence_config)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "wb+") as out:
    io.create_raw_file(out, args.res, args.fps, args.duration)
    if args.format == 'raw':
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            
            for i, frame in enumerate(frame_iterator):
                io.save_frame(out, frame)
                Image.fromarray(np.uint8(frame)).save(f'{tmpdirname}/img_{i}.pgm')
            os.system(f"ffmpeg -f image2 -framerate 30 -i {tmpdirname}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {args.out}")
    else:
        for chunk in formats[args.format].encoder(frame_iterator):
            out.write(chunk)
