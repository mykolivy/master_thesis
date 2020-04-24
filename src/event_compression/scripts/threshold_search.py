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


def seq_provider(res, precision):
	compute_effort = int(math.ceil(1.0 / precision))

	def generate_seq(rate):
		duration = int(math.ceil(compute_effort / (res[0] * res[1])))
		seq_config = Config(res,
		                    1,
		                    duration,
		                    rate=float(rate),
		                    val_range=(0, 256),
		                    dtype='uint8')
		return RandomChange(seq_config)

	return generate_seq


def main():
	args = get_args()

	codec = codecs()[args.codec]()

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w+') as out:
		util.log(f"Parameters used: {args}\n", out)
		util.log(f"RES, DURATION: THRESHOLD", out)
		results = {
		    "res": [0.0 for _ in range(len(args.resolutions))],
		    "dur": [0.0 for _ in range(len(args.durations))]
		}

		util.log("Processing resolutions...", out)
		for i, res in enumerate(args.resolutions):
			seqs = seq_provider((res, res), args.precision)
			results["res"][i] = compute_event_threshold(codec, args.entropy_coder,
			                                            seqs, args.precision)
			util.log(results["res"][i], out)

		util.log('', out)
		util.log(f"RESULT SUMMARY:\n{results}\n", out)


def compute_event_threshold(codec, coder, seqs, precision):
	start = 0.0
	end = 1.0

	bsize = 100
	size = 0

	rate = None

	while abs(bsize - size) >= precision:
		rate = get_pivot(start, end)
		seq = seqs(rate)
		rate = seq.rate

		bsize = compute_baseline_size(coder, seq)
		size = compute_size(codec.encoder(seq))

		print(f"Rate: {rate}, bsize: {bsize}, size: {size}")

		# Adjust interval according to real rate
		if size < bsize:
			start = rate
		else:
			end = rate

		if start > end:
			start, end = end, start

	return rate


def compute_baseline_size(coder: str, seq):
	data = seq_to_bytes(seq)
	with tempfile.NamedTemporaryFile('w+b') as raw:
		with tempfile.NamedTemporaryFile('w+b') as baseline:
			raw.write(data)
			raw.flush()

			os.system(f"{coder} {raw.name} {baseline.name} > /dev/null 2>&1")

			baseline.seek(0, 2)
			return baseline.tell()


def compute_size(encoder):
	return len(functools.reduce(lambda x, y: x + y, encoder, bytearray()))


def get_pivot(start, end):
	return (end - start) / 2.0 + start


@util.log_result()
def get_args():
	parser = util.get_parser(__file__)
	args = parser.parse_args()
	args.entropy_coder = ' '.join(args.entropy_coder.split('~'))
	args.out_redir = '' if args.verbose else '> /dev/null 2>&1'
	return args


def seq_to_bytes(seq):
	result = bytearray()

	for frame in iter(seq):
		assert frame.dtype == 'uint8'
		result += frame.tobytes()

	return result


if __name__ == "__main__":
	main()