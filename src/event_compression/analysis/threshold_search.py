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

			encoded = functools.reduce(operator.add, codec.encoder(seq), bytearray())

			bsize = entropy_size(coder, seq_to_bytes(seq))
			size = entropy_size(coder, encoded)

			yield rate, seq.conf.res, len(seq), bsize, size

			# Adjust interval according to real rate
			if size < bsize:
				start = rate
			else:
				end = rate

			if start > end:
				start, end = end, start
		yield "Precision reached"

	yield res, frames, rate


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


def seq_to_bytes(seq):
	result = bytearray()

	for frame in iter(seq):
		assert frame.dtype == 'uint8'
		result += frame.tobytes()

	return result