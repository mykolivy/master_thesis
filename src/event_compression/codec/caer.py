"""Define CAER-like codecs"""
import numpy as np
import struct
from .util import *
import functools
import itertools


def add_compact_frame(diff, t, arranged):
	for i, row in enumerate(diff):
		for j, value in enumerate(row):
			if value != 0:
				arranged[i][j].append(t)
				if value >= 0:
					arranged[i][j].append(1)
				else:
					arranged[i][j].append(2)


def save_compact_frames(arranged, out):
	for row in arranged:
		for x in row:
			for i in range(0, len(x), 2):
				out.write(x[i].to_bytes(4, byteorder='little', signed=False))
				out.write(x[i + 1].to_bytes(1, byteorder='big', signed=False))
			out.write(int(0).to_bytes(1, byteorder='big', signed=False))


class CAERBinaryDeltaIterator:
	def __init__(self, frame_iterator):
		self.frames = frame_iterator
		self.prev = frame_iterator.start_frame
		self.res = self.frames.conf.res
		self.arranged = None

	def __iter__(self):
		if self.arranged == None:
			self.compute_compact_frames()
		for row in self.arranged:
			for x in row:
				result = bytearray()
				for i in range(0, len(x), 2):
					result += x[i].to_bytes(4, byteorder='little', signed=False)
					result += x[i + 1].to_bytes(1, byteorder='big', signed=False)
				result += int(0).to_bytes(1, byteorder='big', signed=False)
				yield result

	def compute_compact_frames(self):
		self.arranged = [[[] for y in range(self.res[1])]
		                 for x in range(self.res[0])]
		for t, frame in enumerate(self.frames):
			diff = np.subtract(frame, self.prev)
			self.add_compact_frame(diff, t)
			self.prev = frame.copy()

	def add_compact_frame(self, diff, t):
		for i, row in enumerate(diff):
			for j, value in enumerate(row):
				if value != 0:
					self.arranged[i][j].append(t)
					if value >= 0:
						self.arranged[i][j].append(1)
					else:
						self.arranged[i][j].append(2)


class CAERIterator:
	def __init__(self, frame_iterator):
		self.frames = frame_iterator
		self.prev = frame_iterator.start_frame
		self.res = self.frames.conf.res
		self.compute_compact_frames()

	def __iter__(self):
		for row in self.arranged:
			for x in row:
				result = bytearray()
				for i in range(0, len(x), 2):
					result += x[i].to_bytes(4, byteorder='little', signed=False)
					self.append_value(result, x[i + 1])
				result += int(0).to_bytes(1, byteorder='big', signed=True)
				yield result

	def compute_compact_frames(self):
		self.arranged = [[[] for y in range(self.res[1])]
		                 for x in range(self.res[0])]
		for t, frame in enumerate(self.frames):
			diff = np.subtract(frame, self.prev)
			self.add_compact_frame(diff, t)
			self.prev = frame.copy()

	def add_compact_frame(self, diff, t):
		for i, row in enumerate(diff):
			for j, value in enumerate(row):
				if value != 0:
					self.arranged[i][j].append(t)
					self.arranged[i][j].append(value)

	# Converts num to 1-byte or 2-byte signed representation and appends to
	# result. Positive value x is stored as x-1 (0 counts as +1)
	# num != 0, -128 < num < 127
	def append_value(self, result, num):
		if abs(num) >= 127:
			sign = 1 if num > 0 else -1
			result += int(sign * 126).to_bytes(1, byteorder='big', signed=True)
			result += int(num - sign * 128).to_bytes(1, byteorder='big', signed=True)
		else:
			print('single byte')
			result += num.to_bytes(1, byteorder='big', signed=True)


@codec(name="caer")
class CAER:
	"""
	Define encoder and decoder for CAER format.

	The first frame is saved as is: matrix of numbers.
	Then, for each pixel the sequence of events is produced, starting at 
	top-left, moving left-to-right, top-to-bottom.

	CAER format represents each event as:
		value (1 byte), polarity (1 byte), t (4 bytes)

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
		all_events = get_events_by_position(frames)

		# Encode resolution and number of frames
		yield struct.pack('>3I', prev.shape[0], prev.shape[1], len(frames))

		# Encode first frame
		yield prev.tobytes()

		# Yield events for each pixel in turn
		yield from cls._events_to_bytes(all_events)

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		data_it = iter(data)

		header = struct.unpack('>3I', take_next(data_it, 12))
		res = (header[0], header[1])
		frame_num = header[2]

		frame = decode_full_frame(data_it, res)

		# First frame
		yield frame

		# Yield all frames one by one
		all_events = cls._events_from_bytes(data_it, res, frame_num)
		yield from cls._events_to_frames(all_events, frame)

	@classmethod
	def _events_from_bytes(cls, data, res, frame_num):
		"""
		Collect all events from byte stream.

		# Return
		4D list [time][i][j][value1, value2, ...], where values
		"""
		all_events = [np.zeros(res) for t in range(frame_num - 1)]
		for i in range(res[0]):
			for j in range(res[1]):
				events = cls._pixel_events_from_bytes(data)
				for event in events:
					all_events[event[1]][i, j] = event[0]

		return all_events

	@staticmethod
	def _pixel_events_from_bytes(data):
		result = []
		value = take_byte(data)

		while value:
			polarity = take_byte(data)
			if polarity == 0:
				value = -value
			time = take_int(data)

			result.append((value, time))

			value = take_byte(data)

		return result

	@classmethod
	def _events_to_bytes(cls, all_events):
		for row in all_events:
			for events in row:
				result = bytearray()
				for event in events:
					value = event[1]
					cls._pack_event(result, abs(value), get_polarity(value), event[0])
				result += to_bytes(0, 1)
				yield result

	@staticmethod
	def _events_to_frames(events, first_frame):
		frame = first_frame
		for frame_events in events:
			frame = frame_events + frame
			yield frame

	@staticmethod
	def _pack_event(result, value, polarity, time):
		result += to_bytes(value, 1)
		result += to_bytes(polarity, 1)
		result += to_bytes(time, 4)
