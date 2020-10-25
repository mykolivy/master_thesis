"""Define AER-like codecs"""
import numpy as np
import struct
from .util import *
import functools
import itertools
import pdb


@codec(name="aer_to_frames")
class AERToFrames:
	"""
	"""
	@classmethod
	def encoder(cls, frames) -> bytearray:
		"""
		Yield binary representation of events from sequence of video frames
		for single frame at a time.
		"""
		frame_it = iter(frames)
		prev = next(frame_it).copy()

		yield from cls.header(len(frames), prev)

		result = bytearray()
		for (t, i, j, value) in events_from_frames(frames):
			if abs(value) != 0:
				polarity = get_polarity(value)
				cls.append_event(result, i, j, t, abs(value), polarity)
		yield result

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		data_it = iter(data)

		header = struct.unpack('>3I', take_next(data_it, 12))
		res = (header[0], header[1])
		frame_num = header[2]

		frame = decode_full_frame(data_it, res)
		yield frame

		t = 0
		prev_t = 0
		while True:
			try:
				buffer = take_next(data_it, 14)
			except:
				break

			i, j, t, value, polarity = struct.unpack('>3I2B', buffer)

			if t > prev_t:
				for _ in range(t - prev_t):
					yield frame
				prev_t = t

			value = apply_polarity(value, polarity)
			frame[i, j] += value
		yield frame

		for i in range(frame_num - t - 2):
			yield frame

	@staticmethod
	def append_event(result, x, y, t, value, polarity):
		result += to_bytes(x, 4)
		result += to_bytes(y, 4)
		result += to_bytes(t, 4)
		result += to_bytes(value, 1)
		result += to_bytes(polarity, 1)

	@staticmethod
	def header(frame_num, first_frame):
		res = first_frame.shape
		yield struct.pack('>3I', res[0], res[1], frame_num)
		yield first_frame.tobytes()
