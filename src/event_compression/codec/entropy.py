"""Define entropy codecs"""
import struct, tempfile, functools, itertools, os
import numpy as np
from .util import *


@codec(name="lpaq1")
class LPAQ1:
	"""
	"""
	coder = "bin/lpaq1"

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

	@staticmethod
	def seq_to_bytes(seq):
		result = bytearray()

		for frame in iter(seq):
			assert frame.dtype == 'uint8'
			result += frame.tobytes()

		return result
