"""Define AER-like codecs"""
import numpy as np
import struct
from .util import *
import functools
import itertools
import pdb


@codec(name="aer")
class AER:
	"""
	Define encoder and decoder for AER format.

	The first frame is saved as is: matrix of numbers.
	Then the sequence of events is produced.
	
	AER format represents each event as:
			value (1 byte), polarity (1 byte) i (4 bytes), j (4 bytes), t (4 bytes)

	Polarity -- sign of value:
			For positive values: polarity > 0.
			For negative values: polarity = 0.
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


@codec(name="pure_aer")
class PureAER:
	"""
	Define encoder and decoder for Pure AER format.
	Pure AER is the closest format to theoretical AER.
	Each event is only +1 or -1 and only polarity is stored.

	The first frame is saved as is: matrix of numbers.
	Then the sequence of events is produced.
	
	AER format represents each event as:
			i (4 bytes), j (4 bytes), t (4 bytes), polarity (1 byte)

	Polarity -- sign of value:
			For positive values: polarity > 0.
			For negative values: polarity = 0.
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

		for (t, i, j, value) in events_from_frames(iter(frames)):
			if abs(value) != 0:
				result = bytearray()
				polarity = get_polarity(value)
				for _ in range(abs(value)):
					cls.append_event(result, i, j, t, polarity)
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
				buffer = take_next(data_it, 13)
			except:
				break

			i, j, t, polarity = struct.unpack('>3I1B', buffer)

			if t > prev_t:
				for _ in range(t - prev_t):
					yield frame
				prev_t = t

			value = apply_polarity(1, polarity)
			frame[i, j] += value
		yield frame

		for i in range(frame_num - t - 2):
			yield frame

	@staticmethod
	def append_event(result, x, y, t, polarity):
		result += to_bytes(x, 4)
		result += to_bytes(y, 4)
		result += to_bytes(t, 4)
		result += to_bytes(polarity, 1)

	@staticmethod
	def header(frame_num, first_frame):
		res = first_frame.shape
		yield struct.pack('>3I', res[0], res[1], frame_num)
		yield first_frame.tobytes()


@codec(name="aer_rgb")
class RGBAER:
	"""
	Define encoder and decoder for rgb color AER format.

	The first frame is saved as is: matrix of numbers.
	Then the sequence of events is produced.
	
	AER format represents each event as:
			value (1 byte), polarity (1 byte) i (4 bytes), j (4 bytes), t (4 bytes)

	Polarity -- sign of value:
			For positive values: polarity > 0.
			For negative values: polarity = 0.
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
		for t, i, j, *values in events_from_vector_frames(frames):
			cls.append_event(result, i, j, t, values)
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
	def append_event(result, x, y, t, values):
		result += to_bytes(x, 4)
		result += to_bytes(y, 4)
		result += to_bytes(t, 4)
		for value in values:
			result += to_bytes(abs(int(value)), 1)
			result += to_bytes(get_polarity(value), 1)

	@staticmethod
	def header(frame_num, first_frame):
		res = first_frame.shape
		yield struct.pack('>3I', res[0], res[1], frame_num)
		yield first_frame.tobytes()


@codec(name="aer_lossy")
class AERLossy:
	"""
	Same as AER format, but lossy.

	AER format represents each event as:
			i (4 bytes), j (4 bytes), t (float), value (1 byte), polarity (1 byte)

	Events with value smaller or equal to the threshold are ignored.
	"""
	@classmethod
	def encoder(cls, frames, threshold=5) -> bytearray:
		"""
		Yield binary representation of events from sequence of video frames
		for single frame at a time.
		"""
		frame_it = iter(frames)
		prev = next(frame_it).copy()

		yield from AER.header(len(frames), prev)

		for t, frame in enumerate(frame_it):
			diff = np.subtract(frame, prev, dtype='int16')
			result = bytearray()
			for (i, j), value in np.ndenumerate(diff):
				value = int(value)
				if abs(value) <= threshold:
					continue

				polarity = get_polarity(value)
				AER.append_event(result, i, j, t, abs(value), polarity)

			to_update = abs(diff) > threshold
			prev[to_update] = frame[to_update]
			if len(result) != 0:
				yield result

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		yield from AER.decoder(data)


@codec(name="aer_lossy_accumulated")
class AERLossyAccumulated:
	"""
	Same as AER format, but lossy.

	AER format represents each event as:
			i (4 bytes), j (4 bytes), t (float), value (1 byte), polarity (1 byte)

	Events with value smaller or equal to the threshold are ignored.
	Consecutive ignored values are accumulated, until their sum is > threshold, in
	which case a new event with value of this sum is created at the current
	timestamp.
	"""
	@classmethod
	def encoder(cls, frames, threshold=50) -> bytearray:
		"""
		Yield binary representation of events from sequence of video frames
		for single frame at a time.
		"""
		frame_it = iter(frames)
		prev = next(frame_it).copy()

		yield from AER.header(len(frames), prev)

		result = bytearray()
		for (t, i, j, value) in events_from_frames(frames):
			if abs(value) > threshold:
				polarity = get_polarity(value)
				AER.append_event(result, i, j, t, abs(value), polarity)
		yield result

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		yield from AER.decoder(data)
