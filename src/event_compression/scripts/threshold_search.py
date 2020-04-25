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


def seq_provider(res, precision):
	compute_effort = int(math.ceil(1.0 / precision))

	def generate_seq(rate):
		duration = int(math.ceil(compute_effort / (res[0] * res[1])))
		duration = max(duration, 2)
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
		results = {
		    "res": [0.0 for _ in range(len(args.resolutions))],
		    "dur": [0.0 for _ in range(len(args.durations))]
		}
		util.log("Processing resolutions...", out)
		util.log("{0:^20} {1:^20}".format("Resolution", "Threshold"), out)
		util.log("{0:=^41}".format(''), out)
		for i, res in enumerate(args.resolutions):
			seqs = seq_provider((res, res), args.precision)
			util.log("{0:^19} {1:^10} {2:^10}".format("Rate", "bsize", "size"), out)
			util.log("{0:-^41}".format(''), out)

			results["res"][i] = compute_event_threshold(codec, args.entropy_coder,
			                                            seqs, args.precision, out)

			util.log("{0:=^41}".format(''), out)
			util.log("{0:^20} {1:^20.8f}".format(res, results["res"][i]), out)
			util.log("{0:=^41}".format(''), out)


def compute_event_threshold(codec, coder, seqs, precision, out):
	start = 0.0
	end = 1.0

	bsize = 0
	size = 1

	while bsize < size:
		start = start - 0.01
		rate = 0.0
		prev_rate = 1.0
		while abs(rate - prev_rate) >= precision:
			prev_rate = rate
			rate = get_pivot(start, end)
			seq = seqs(rate)
			rate = seq.rate

			encoded = functools.reduce(operator.add, codec.encoder(seq), bytearray())

			bsize = entropy_size(coder, seq_to_bytes(seq))
			size = entropy_size(coder, encoded)

			util.log("{0:^19.15f} {1:^10} {2:^10}".format(rate, bsize, size), out)

			# Adjust interval according to real rate
			if size < bsize:
				start = rate
			else:
				end = rate

			if start > end:
				start, end = end, start
		print("{0:^41}".format("Precision reached"))

	return rate


def entropy_size(coder: str, data):
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