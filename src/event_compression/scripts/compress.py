#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import functools
from event_compression.scripts import util
from event_compression.codec import codecs
from event_compression.sequence.synthetic import RandomChange, Config
import math
import tempfile
import operator
import json
from pathlib import Path
from event_compression.sequence.video import VideoSequence, FileBytes


def compress(codec, inp, out):
	frames = VideoSequence(inp)
	for data in codec.encoder(frames):
		out.write(data)


def decompress(codec, inp, out):
	compr_data = FileBytes(inp)
	for data in codec.decoder(compr_data):
		out.write(data)


def main():
	parser = util.get_parser(__file__)
	args = parser.parse_args()
	codec = codecs()[args.codec]

	with open(args.input, 'rb') as inp:
		with open(args.output, 'wb+') as out:
			if args.decompress:
				decompress(codec, inp, out)
			else:
				compress(codec, inp, out)

			inp_size = os.path.getsize(inp.name)
			out_size = os.path.getsize(out.name)
			print(f"Compressed {inp_size} --> {out_size}")


if __name__ == '__main__':
	main()
