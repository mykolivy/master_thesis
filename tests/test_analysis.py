import event_compression.analysis.event as ev
import numpy as np


def test_event_rate():
	seq = [
	    np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
	             dtype='uint8'),
	    np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
	             dtype='uint8'),
	    np.array([[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
	             dtype='uint8'),
	    np.array([[3, 3, 3, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
	             dtype='uint8')
	]
	result = ev.event_rate(seq)
	assert result == 0.25