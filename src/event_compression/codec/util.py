import numpy as np
from . import _REGISTER
from functools import reduce
import tempfile
from os import system


def codecs():
	return _REGISTER.copy()


def codec(name=None):
	def decorate(cls):
		_REGISTER[name] = cls
		return cls

	return decorate


def create_raw_file(f, resolution, fps, duration):
	f.write((resolution[0]).to_bytes(4, byteorder='big'))
	f.write((resolution[1]).to_bytes(4, byteorder='big'))
	f.write((fps).to_bytes(1, byteorder='big'))
	f.write((duration).to_bytes(4, byteorder='big'))


def save_frame(out, data):
	for row in np.uint8(data):
		for j in row:
			out.write(int(j).to_bytes(1, byteorder='big'))


def save_event_frame(diff, t, out):
	for i, row in enumerate(diff):
		for j, value in enumerate(row):
			if value != 0:
				out.write(i.to_bytes(4, byteorder='big', signed=False))
				out.write(j.to_bytes(4, byteorder='big', signed=False))
				out.write(t.to_bytes(4, byteorder='little', signed=False))
				if value >= 0:
					out.write(int(1).to_bytes(1, byteorder='big', signed=False))
				else:
					out.write(int(2).to_bytes(1, byteorder='big', signed=False))


def events_from_frames(frames):
	"""
	Generate events one-by-one from frames.
	"""
	frame_it = iter(frames)
	prev = next(frame_it).copy()

	for t, frame in enumerate(frame_it):
		diff = np.subtract(frame, prev, dtype='int16')
		for (i, j), value in np.ndenumerate(diff):
			value = int(value)
			if value != 0:
				yield (t, i, j, value)
		prev = frame.copy()


def events_from_vector_frames(frames):
	"""
	Generate events one-by-one from frames.
	"""
	frame_it = iter(frames)
	prev = next(frame_it).copy()

	for t, frame in enumerate(frame_it):
		diff = np.subtract(frame, prev, dtype='int16')
		for index in np.ndindex(*diff.shape[:2]):
			if not all(diff[index] == 0):
				yield (t, index[0], index[1], *diff[index])
		prev = frame.copy()


def get_events_by_position(frames, frame_num=None):
	"""
	Collect events from frames, grouped by their coordinates.

	# Arguments
		frames: iterable of np.ndarray, frames of video sequence from which to 
		  extract event information.
		frame_num: int > 0, number of first elements from `frames` to take into
		  account. If `not frame_num` then the entire sequence is
			considered
	
	# Returns
		3D-list of events. 
		First two dimensions signify position of the event.
		Last dimension is a list of tuples (time, value).

		Time is the index of the frame.
		Value is change in brightness.
	"""
	frames_it = iter(frames)
	first_frame = next(frames_it)
	res = first_frame.shape

	events = [[[] for col in range(res[1])] for row in range(res[0])]

	prev = first_frame.copy()
	for t, frame in enumerate(frames_it):
		#assert id(frame) != id(prev)
		diff = np.subtract(frame, prev, dtype='int16')

		for (i, j), value in np.ndenumerate(diff):
			value = int(value)
			if value == 0:
				continue
			events[i][j].append((t, value))

		prev = frame.copy()

	return events


def get_polarity(value):
	"""
	Return polarity of the value.

	Value <  0 -> return 0
	Value >= 0 -> return 1
	"""
	return 0 if value < 0 else 1


def apply_polarity(value, polarity):
	"""
	Combine value and polarity.

	# Arguments
		value: int >= 0, Absolute value.
		polarity: int, Value indicating the sign.
			Value is considered to be negative when `polarity == 0`.
			Any non-zero polarity indicates positive value.
	
	# Return
	Integer with absolute value of `value` and sign indicated by `polarity`.
	"""
	return value if polarity != 0 else -value


def to_bytes(value, bytes_num, order='big', signed=False) -> bytes:
	"""
	Convert value to bytes representation.

	# Arguments
		value: int, Value to be converted.
		bytes_num: int > 0, Number of bytes in representation of `value`.
			Must be sufficiently big to store `value`
		order: 'big' or 'small', Ordering of bytes in representation.
		singed: Boolean, Wether to represent `value` as a signed number.
	"""
	return value.to_bytes(bytes_num, byteorder=order, signed=signed)


def decode_full_frame(data, res) -> np.ndarray:
	"""
	Extract full frame from stream of bytes.

	Reads enough bytes to fill `res` frame.

	# Arguments
		data: bytes-like, Stream of bytes at beggining of a frame values.
		res: (int > 0, int > 0), Resolution of the frame.

	# Return
		Decoded frame.
	"""
	result = np.zeros(res, dtype='uint8')
	for index, _ in np.ndenumerate(result):
		result[index] = next(data)
	return result


def take_next(data, n) -> bytearray:
	"""
	Read next `n` bytes from source.

	# Arguments
		data: bytes-like, Stream of bytes to read from.
		n: int > 0, Number of bytes to read.
	
	# Return
		`n` next bytes from `data`
	"""
	it = iter(data)
	result = bytearray()
	for _ in range(n):
		result += to_bytes(next(it), 1)
	return result


def take_byte(data) -> int:
	"""
	Read next byte form `data` as an `int`.

	# Arguments
		data: bytes-like, Stream of bytes to read from.

	# Return
		`int` representation of the next byte in `data`.
	"""
	return int.from_bytes(take_next(data, 1), byteorder='big')


def take_int(data) -> int:
	"""
	Read next 4 bytes from `data` as an `int`.

	# Arguments
		data: bytes-like, Stream of bytes to read from.

	# Return
		`int` representation of the next 4 bytes in `data`.
	"""
	return int.from_bytes(take_next(data, 4), byteorder='big')