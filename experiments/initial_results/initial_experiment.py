from PIL import Image
import numpy as np
import sys
import os
import argparse
import tempfile
from pathlib import Path
import event_compression.sequence as sequence
import event_compression.sequence.synthetic as synthetic
import event_compression.codec as codec
from event_compression.scripts import util
import pandas as pd
import os, sys

names = ["moving_edge", "single_color", "random_pixel", "checkers"]
# formats = ["aer", "caer", "raw"]
formats = ["short_aer"]

codecs = codec.codecs()
sequences = sequence.sequences()

entropy_coder = sys.argv[1]


def entropy_compress(path, out_path):
	command = f"{entropy_coder} {path} {out_path}"
	print(f"Running: {command}")
	os.system(command)


def get_frame_iterator(name):
	sequence_config = synthetic.Config([256, 256],
	                                   30,
	                                   5,
	                                   value=0,
	                                   rate=0.6,
	                                   val_range=[0, 255])
	return sequences[name](sequence_config)


for name in names:
	print(f"Processing {name}...")
	frame_it = get_frame_iterator(name)
	frames = [x.copy() for x in frame_it]

	for frm in formats:
		file_name = f"data/{name}.{frm}"
		with open(file_name, "wb+") as out:
			if frm == "raw":
				with tempfile.TemporaryDirectory() as tmpdirname:
					for i, frame in enumerate(frames):
						codec.util.save_frame(out, frame)
						Image.fromarray(np.uint8(frame)).save(f'{tmpdirname}/img_{i}.pgm')
					os.system(
					    f"ffmpeg -f image2 -framerate 30 -i {tmpdirname}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray data/{name}.mp4"
					)
			else:
				encoder = codecs[frm].encoder(frames)
				for chunk in encoder:
					out.write(chunk)
		entropy_compress(file_name, f"{file_name}.paq")