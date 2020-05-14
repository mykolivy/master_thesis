"""Define entropy codecs"""
import struct, tempfile, functools, itertools, os
from pathlib import Path
import numpy as np
from .util import *


@codec(name="lpaq1")
class LPAQ1:
	"""
	"""
	coder = Path(__file__).parents[1] / Path("scripts/bin/lpaq1")

	@classmethod
	def encoder(cls, frames, N=0) -> bytearray:
		data = cls.seq_to_bytes(frames)
		with tempfile.NamedTemporaryFile('w+b') as raw:
			with tempfile.NamedTemporaryFile('w+b') as encoded:
				raw.write(data)
				raw.flush()

				os.system(f"{cls.coder} {N} {raw.name} {encoded.name}")

				with open(encoded.name, "rb") as f:
					yield f.read()

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		with tempfile.NamedTemporaryFile('w+b') as encoded:
			with tempfile.NamedTemporaryFile('w+b') as decoded:
				encoded.write(data)
				encoded.flush()

				os.system(f"{cls.coder} d {encoded.name} {decoded.name}")

				with open(decoded.name, "rb") as f:
					yield f.read()

	@classmethod
	def encode(cls, in_path, out_path, N=0):
		os.system(f"{cls.coder} {N} {in_path} {out_path}")

	@classmethod
	def decode(cls, in_path, out_path):
		os.system(f"{cls.coder} d {in_path} {out_path}")

	@staticmethod
	def seq_to_bytes(seq):
		result = bytearray()

		for frame in iter(seq):
			assert frame.dtype == 'uint8'
			result += frame.tobytes()

		return result


@codec(name="raw")
class RAW:
	@classmethod
	def encoder(cls, frames) -> bytearray:
		it = iter(frames)

		length = len(frames)
		if length > 0:
			first_frame = next(it)
			res = first_frame.shape

			yield struct.pack('>3I', res[0], res[1], length)
			yield first_frame.tobytes()

			for frame in it:
				yield frame.tobytes()

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		data_it = iter(data)

		header = struct.unpack('>3I', take_next(data_it, 12))
		res = (header[0], header[1])
		frame_num = header[2]

		for i in range(frame_num):
			yield decode_full_frame(data_it, res)


@codec(name="residual")
class Residual:
	@classmethod
	def encoder(cls, frames) -> bytearray:
		it = iter(frames)

		length = len(frames)
		if length > 0:
			first_frame = next(it)
			res = first_frame.shape

			yield struct.pack('>3I', res[0], res[1], length)
			yield first_frame.tobytes()

			prev = first_frame.copy()
			for frame in it:
				diff = np.subtract(frame, prev, dtype='int16')
				yield diff.tobytes()
				prev = frame.copy()

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		data_it = iter(data)

		header = struct.unpack('>3I', take_next(data_it, 12))
		res = (header[0], header[1])
		frame_num = header[2]

		for i in range(frame_num):
			yield decode_full_frame(data_it, res)
