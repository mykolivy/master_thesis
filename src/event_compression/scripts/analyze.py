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
from event_compression.analysis.event import event_rate
from event_compression.sequence.video import VideoSequence, FileBytes, GrayscaleVideoConverter


def analyze(video_path):
	seq = GrayscaleVideoConverter(VideoSequence(video_path))
	return event_rate(seq)


def main():
	parser = util.get_parser(__file__)
	args = parser.parse_args()

	results = []

	for f in args.files:
		paths = Path().glob(f)
		for p in paths:
			result = analyze(p)
			print(f"{p} --> rate: {result}")
			results.append(result)

	print(results)


if __name__ == '__main__':
	main()
