"""Define CAER-like codecs"""
import numpy as np
import struct
from .util import *
import functools
import itertools


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
		all_events = cls.events_from_bytes(data_it, res, frame_num)
		yield from cls.events_to_frames(all_events, frame)

	@classmethod
	def events_from_bytes(cls, data, res, frame_num):
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
					cls.pack_event(result, abs(value), get_polarity(value), event[0])
				result += to_bytes(0, 1)
				yield result

	@staticmethod
	def events_to_frames(events, first_frame):
		frame = first_frame
		for frame_events in events:
			frame = frame_events + frame
			yield frame

	@staticmethod
	def pack_event(result, value, polarity, time):
		result += to_bytes(value, 1)
		result += to_bytes(polarity, 1)
		result += to_bytes(time, 4)


@codec(name="caer_lossy")
class CAERLossy:
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
	def encoder(cls, frames, threshold=5) -> bytearray:
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
		yield from cls._events_to_bytes(all_events, threshold)

	@classmethod
	def decoder(cls, data) -> np.ndarray:
		yield from CAER.decoder(data)

	@classmethod
	def _events_to_bytes(cls, all_events, threshold):
		for row in all_events:
			for events in row:
				result = bytearray()
				s = 0
				for event in events:
					s += event[1]
					if abs(s) > threshold:
						CAER.pack_event(result, abs(s), get_polarity(s), event[0])
						s = 0
				result += to_bytes(0, 1)
				yield result
