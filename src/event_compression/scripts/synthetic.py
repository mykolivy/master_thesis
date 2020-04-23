#!/usr/bin/env python3

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


def main():
	formats = codec.codecs()
	formats['raw'] = None  # Register additional format

	# Define script interface
	args = get_args()

	frame_iterator = get_frame_iterator(args)

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "wb+") as out:
		codec.util.create_raw_file(out, args.res, args.fps, args.duration)
		if args.format == 'raw':
			with tempfile.TemporaryDirectory() as tmpdirname:
				for i, frame in enumerate(frame_iterator):
					codec.util.save_frame(out, frame)
					Image.fromarray(np.uint8(frame)).save(f'{tmpdirname}/img_{i}.pgm')
				os.system(
				    f"ffmpeg -f image2 -framerate 30 -i {tmpdirname}/img_%d.pgm -c:v libx264 -preset veryslow -crf 0 -pix_fmt gray {args.out}"
				)
		else:
			for chunk in formats[args.format].encoder(frame_iterator):
				out.write(chunk)


@util.log_result()
def get_args():
	parser = util.get_parser(__file__)
	return parser.parse_args()


def get_frame_iterator(args):
	seqs = sequence.sequences()
	sequence_config = synthetic.Config(args.res,
	                                   args.fps,
	                                   args.duration,
	                                   value=args.value,
	                                   rate=args.rate,
	                                   val_range=args.range)
	return seqs[args.sequence](sequence_config)


if __name__ == "__main__":
	main()