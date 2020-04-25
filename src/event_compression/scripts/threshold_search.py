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


def res_seq_provider(res, compute_effort):
	res = (res, res)

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


def time_seq_provider(time, compute_effort):
	assert time > 1

	def generate_seq(rate):
		res = int(math.ceil(math.sqrt(compute_effort / time)))
		res = max(res, 1)
		seq_config = Config((res, res),
		                    1,
		                    time,
		                    rate=float(rate),
		                    val_range=(0, 256),
		                    dtype='uint8')
		return RandomChange(seq_config)

	return generate_seq


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

			util.log("{0:^27.15f} {1:^10} {2:^10}".format(rate, bsize, size), out)

			# Adjust interval according to real rate
			if size < bsize:
				start = rate
			else:
				end = rate

			if start > end:
				start, end = end, start
		print("{0:^49}".format("Precision reached"))

	return rate


def entropy_size(coder: str, data):
	with tempfile.NamedTemporaryFile('w+b') as raw:
		with tempfile.NamedTemporaryFile('w+b') as baseline:
			raw.write(data)
			raw.flush()

			os.system(f"{coder} {raw.name} {baseline.name} > /dev/null 2>&1")

			baseline.seek(0, 2)
			return baseline.tell()


def get_pivot(start, end):
	return (end - start) / 2.0 + start


def get_args():
	parser = util.get_parser(__file__)
	args = parser.parse_args()
	args.entropy_coder = ' '.join(args.entropy_coder.split('~'))
	args.out_redir = '' if args.verbose else '> /dev/null 2>&1'
	return args


def print_args(args, out):
	for key, value in vars(args).items():
		util.log("{0:<30} {1}".format(key + ":", value), out)


def seq_to_bytes(seq):
	result = bytearray()

	for frame in iter(seq):
		assert frame.dtype == 'uint8'
		result += frame.tobytes()

	return result


def tabulate_search(points, seq_provider, args, out, header):
	compute_effort = int(math.ceil(1.0 / args.precision)) * args.compute_effort
	codec = codecs()[args.codec]()
	results = [0.0 for _ in range(len(points))]

	util.log("{0:=^49}".format(''), out)
	util.log("({0:^5}) {1:^20} {2:^20}".format("#", header, "Threshold"), out)
	util.log("{0:=^49}".format(''), out)

	for i, point in enumerate(points):
		samples = []
		for j in range(args.iterations):

			seqs = seq_provider(point, compute_effort)
			util.log("{0:^27} {1:^10} {2:^10}".format("Rate", "bsize", "size"), out)
			util.log("{0:-^49}".format(''), out)

			threshold = compute_event_threshold(codec, args.entropy_coder, seqs,
			                                    args.precision, out)
			samples.append(threshold)

			util.log("{0:=^49}".format(''), out)
			util.log("({0:^5}) {1:^20} {2:^20.8f}".format(j + 1, point, threshold),
			         out)
			util.log("{0:=^49}".format(''), out)

		results[i] = sum(samples) / args.iterations
		util.log("{0:*^49}".format(''), out)
		util.log("{0} {1:^20} {2:^20.8f}".format("average", point, results[i]), out)
		util.log("{0:*^49}".format(''), out)

	util.log("{0:^49}".format("SUMMARY"), out)
	util.log("{0:^24} {1:^24}".format(header, "Threshold"), out)
	util.log("{0:=^49}".format(''), out)
	for i, point in enumerate(points):
		util.log("{0:^24} {1:^24.15f}".format(point, results[i]), out)
	util.log("{0:=^49}".format(''), out)


def main():
	args = get_args()

	if os.path.isfile(args.out):
		sys.stderr.write(f"ERROR: file {args.out} already exists! Exiting...")
		exit(1)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w+') as out:
		print_args(args, out)

		util.log("\nProcessing resolutions...", out)
		tabulate_search(args.resolutions, res_seq_provider, args, out, "Resolution")

		util.log("\nProcessing durations...", out)
		tabulate_search(args.durations, time_seq_provider, args, out, "Duration")


if __name__ == "__main__":
	main()