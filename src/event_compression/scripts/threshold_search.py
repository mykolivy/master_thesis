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
from event_compression.analysis.threshold_search import *
import math
import tempfile
import operator
import json


def get_args():
	parser = util.get_parser(__file__)
	args = parser.parse_args()
	args.entropy_coder = ' '.join(args.entropy_coder.split('~'))
	args.out_redir = '' if args.verbose else '> /dev/null 2>&1'
	return args


def print_args(args, out):
	for key, value in vars(args).items():
		util.log("{0:<30} {1}".format(key + ":", value), out)


def tabulate_search(points, seq_provider, args, out):

	codec = codecs()[args.codec]()
	results = [None for _ in range(len(points))]
	table = Table(80, lambda x: util.log(x, out))

	table.print_line("=")
	table.print("#.",
	            "Resolution",
	            "Frames",
	            "Threshold",
	            lengths=[5, 20, 20, 35])
	table.print_line("=")

	for i, point in enumerate(points):
		samples = []
		threshold = (None, None, None, None, None)
		for j in range(args.iterations):

			seqs = seq_provider(*point, val_range=args.range)
			table.print("Rate", "Resolution", "Frames", "bsize", "size")
			table.print_line("-")

			tab = tab_event_threshold(codec, args.entropy_coder, seqs, args.precision)
			threshold = get_threshold(tab, table)
			samples.append(threshold[2])

			table.print_line("=")
			table.print(f"{j+1}.", *threshold, lengths=[5, 20, 20, 35])
			table.print_line("=")

		results[i] = (threshold[0], threshold[1], np.mean(samples))
		table.print_line("*")
		table.print("avg:", *results[i], lengths=[5, 25, 25, 25])

		# Bootstrap for confidence interval
		B = 100000
		bsamples = np.random.choice(samples, (len(samples), B))
		bmeans = np.mean(bsamples, axis=0)
		bmeans.sort()
		bmean = np.mean(bmeans)
		conf_int = np.percentile(bmeans, [2.5, 97.5])
		table.print(f"Bootstrap mean: {bmean}")
		table.print(f"Confidence interval (95%): {conf_int}")
		table.print(f"Samples: {samples}")
		table.print_line("*")

	table.print("SUMMARY")
	table.print("Resolution", "Frames", "Threshold", lengths=[20, 20, 40])
	table.print_line("=")
	for i, point in enumerate(points):
		table.print(*results[i], lengths=[20, 20, 40])
	table.print_line("=")


def get_threshold(event_threshold_tab, table):
	it = iter(event_threshold_tab)
	result = None
	prev = next(it, None)
	while prev:
		result = prev
		prev = next(it, None)
		if prev:
			if isinstance(result, str):
				table.print(result)
			else:
				table.print(*result)
	return result


class Table:
	def __init__(self, width, printer):
		self.width = width
		self.printer = printer

	def print(self, *args, lengths=None):
		pattern = ""
		args = [arg for arg in args]

		if not lengths:
			length = int(self.width / len(args))
			lengths = [length for x in args]

		if len(args) != len(lengths) or sum(lengths) != self.width:
			breakpoint()
			raise AttributeError()

		pattern = ""
		for i, arg in enumerate(args):
			if isinstance(arg, tuple):
				args[i] = str(arg)

		for arg, length in zip(args, lengths):
			if isinstance(arg, float):
				pattern = pattern + ("{:^" + f"{length}" + ".10}")
			else:
				pattern = pattern + ("{:^" + f"{length}" + "}")

		self.printer(pattern.format(*args))

	def print_line(self, symbol):
		pattern = "{0:" + f"{symbol}" + "^" + f"{self.width}" + "}"
		self.printer(pattern.format(""))


def main():
	args = get_args()
	args.range = tuple(args.range)

	if os.path.isfile(args.out):
		sys.stderr.write(f"ERROR: file {args.out} already exists! Exiting...")
		exit(1)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w+') as out:
		print_args(args, out)

		if args.mode == 'compute_load':
			comp_eff = int(math.ceil(1.0 / args.precision)) * args.compute_effort
			comp_effs = [comp_eff] * len(args.resolutions)

			util.log("\n\nProcessing resolutions...", out)
			points = list(zip(args.resolutions, comp_effs))
			tabulate_search(points, res_seq_provider, args, out)

			comp_effs = [comp_eff] * len(args.durations)

			util.log("\n\nProcessing durations...", out)
			points = list(zip(args.durations, comp_effs))
			tabulate_search(points, time_seq_provider, args, out)
		else:
			for duration in args.durations:

				table = Table(80, lambda x: util.log(x, out))
				table.print_line("")
				table.print_line("~")
				table.print(f"FRAME NUMBER: {duration}")
				table.print_line("~")
				points = list(zip(args.resolutions, [duration] * len(args.resolutions)))
				tabulate_search(points, res_time_seq_provider, args, out)


if __name__ == "__main__":
	main()