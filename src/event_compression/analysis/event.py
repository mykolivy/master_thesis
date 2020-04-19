import numpy as np


def event_rate(sequence):
	"""
	Compute average event rate of the sequence.

	# Arguments
		sequence: iterable of np.ndarray, sequence of frames.
	
	# Return
		float in [0.0, 1.0], average event rate of the sequence.
	"""
	it = iter(sequence)
	prev = next(it).copy()
	shape = prev.shape
	s = 0
	for frame in it:
		s += np.count_nonzero(frame != prev)
		prev = frame.copy()

	total = (len(sequence) - 1) * shape[0] * shape[1]

	return s / total