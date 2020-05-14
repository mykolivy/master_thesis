#!/usr/bin/env python3

import numpy as np
import functools
from event_compression.scripts import util
from event_compression.codec import codecs
from event_compression.sequence.synthetic import RandomChange, Config
import math
import operator
import tempfile
import os


def res_seq_provider(res, compute_effort, val_range=(0, 256)):
	res = (res, res)

	def generate_seq(rate):
		duration = int(math.ceil(compute_effort / (res[0] * res[1])))
		duration = max(duration, 2)
		seq_config = Config(res,
		                    1,
		                    duration,
		                    rate=float(rate),
		                    val_range=val_range,
		                    dtype='uint8')
		return RandomChange(seq_config)

	return generate_seq


def time_seq_provider(time, compute_effort, val_range=(0, 256)):
	assert time > 1

	def generate_seq(rate):
		res = int(math.ceil(math.sqrt(compute_effort / time)))
		res = max(res, 1)
		seq_config = Config((res, res),
		                    1,
		                    time,
		                    rate=float(rate),
		                    val_range=val_range,
		                    dtype='uint8')
		return RandomChange(seq_config)

	return generate_seq


def res_time_seq_provider(res, frame_num, val_range=(0, 256)):
	assert frame_num > 1

	def generate_seq(rate):
		seq_config = Config((res, res),
		                    1,
		                    frame_num,
		                    rate=float(rate),
		                    val_range=val_range,
		                    dtype='uint8')
		return RandomChange(seq_config)

	return generate_seq


def tab_event_threshold(codec, coder, seqs, precision):
	start = 0.0
	end = 1.0

	bsize = 0
	size = 1
	seq = seqs(0.0)
	res = seq.conf.res
	frames = len(seq)

	while bsize < size:
		start = start - 0.01
		rate = 0.0
		prev_rate = 1.0
		while abs(rate - prev_rate) >= precision:
			prev_rate = rate
			rate = get_pivot(start, end)
			seq = seqs(rate)
			rate = seq.rate

			bsize = 0.
			size = 0.

			if coder == "entropy":
				bsize = compute_entropy(seq_to_bytes(seq))
				size = compute_entropy(codec.encoder(seq))
			else:
				encoded = functools.reduce(operator.add, codec.encoder(seq),
				                           bytearray())
				raw = bytearray()
				for frame in iter(seq):
					assert frame.dtype == 'uint8'
					raw += frame.tobytes()

				if coder == "entropy_size":
					bsize = compute_entropy(seq_to_bytes(seq)) * len(raw)
					size = compute_entropy(codec.encoder(seq)) * len(encoded)
				else:
					bsize = compute_size(coder, raw)
					size = compute_size(coder, encoded)

			yield rate, seq.conf.res, len(seq), bsize, size

			# Adjust interval according to real rate
			if size < bsize:
				start = rate
			else:
				end = rate

			if start > end:
				start, end = end, start
		yield "Precision reached"
		if rate == 0:
			break

	yield res, frames, rate


def compute_size(coder: str, data):
	with tempfile.NamedTemporaryFile('w+b') as raw:
		with tempfile.NamedTemporaryFile('w+b') as baseline:
			raw.write(data)
			raw.flush()

			os.system(f"{coder} {raw.name} {baseline.name} > /dev/null 2>&1")

			baseline.seek(0, 2)
			return baseline.tell()


def compute_entropy(data):
	histogram = np.array([0.0 for x in range(256)])
	for chunk in data:
		for b in chunk:
			histogram[b] += 1
	histogram = histogram / np.sum(histogram)

	entropy = 0.0
	for p in histogram:
		if p > 0:
			entropy -= p * math.log(p, 256)
	return entropy


def get_pivot(start, end):
	return max((end - start) / 2.0 + start, 0)


def seq_to_bytes(seq):
	for frame in iter(seq):
		assert frame.dtype == 'uint8'
		yield frame.tobytes()