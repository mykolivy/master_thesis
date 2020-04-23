import pytest
import event_compression.sequence as sequence
from event_compression.sequence import sequences
from event_compression.sequence.synthetic import *
import event_compression.codec as codec
from event_compression.codec import caer as caer
import itertools
import functools
import numpy as np
import pdb
import struct
import copy


def simple_seq():
	"""
	Produce sequence with events (x, y, t, v, p):
			(1, 1, 1, 255, +)
			(1, 1, 3, 248, -)
			(0, 0, 4, 255, +)
			(1, 1, 4, 7,   -)
	"""
	return [
	    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8'),
	    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8'),
	    np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype='uint8'),
	    np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype='uint8'),
	    np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype='uint8'),
	    np.array([[255, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8')
	]


def reduce(encoder):
	return functools.reduce(lambda x, y: x + y, encoder, bytearray())


def invertability_test(cls=None, threshold=0):
	"""
    Define common tests for all aer codecs.

    @common_test(cls=aer_class_to_be_tested)
    class TestClass:
        pass
    """
	def decorate(_cls):
		_cls.cls = cls

		def test_invertability(instance, sequence, correct):
			code = reduce(instance.cls.encoder(sequence))
			decoder = instance.cls.decoder(code)
			frames = [x.copy() for x in decoder]
			for t, f in enumerate(frames):
				for index, value in np.ndenumerate(f):
					assert abs(int(value) - int(correct[t][index])) <= threshold

		#Add all invertability tests
		for name, sequence in sequences().items():
			config = Config((64, 64), 10, 3, value=127, rate=0.2)
			np.random.seed(seed=0)
			correct_seq = [x.copy() for x in sequence(config)]
			seq = copy.deepcopy(correct_seq)

			def caller(seq, correct_seq):
				return lambda inst: test_invertability(inst, seq, correct_seq)

			setattr(_cls, f'test_invertability_{name}', caller(seq, correct_seq))
		fn = lambda inst: test_invertability(inst, simple_seq(), simple_seq())
		setattr(_cls, f'test_invertability_simple', fn)

		return _cls

	return decorate


def test_init():
	assert codec.codecs()


@invertability_test(caer.CAER)
class TestCAER:
	def test_encoder(self):
		sequence = simple_seq()
		code = reduce(self.cls.encoder(sequence))
		correct_code = struct.pack('>3I 9B BBIB B B B BBIBBIBBIB B B B B', 3, 3, 6,
		                           0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 4, 0, 0, 0, 0,
		                           255, 1, 1, 248, 0, 3, 7, 0, 4, 0, 0, 0, 0, 0)
		assert code == correct_code

	def test_encode_moving_edge(self):
		seq = MovingEdge(Config((3, 3), 2, 2))
		code = reduce(self.cls.encoder(seq))
		correct_code = struct.pack('>3I 9B BBIB BBIB B BBIB BBIB B BBIB BBIB B', 3,
		                           3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 0, 0,
		                           255, 1, 1, 0, 0, 255, 1, 0, 0, 255, 1, 1, 0, 0,
		                           255, 1, 0, 0, 255, 1, 1, 0, 0)
		assert code == correct_code

	def test_encode_single_color(self):
		seq = SingleColor(Config((3, 3), 4, 1, value=127))
		code = reduce(self.cls.encoder(seq))
		correct_code = struct.pack('>3I 18B', 3, 3, 4, 127, 127, 127, 127, 127, 127,
		                           127, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		assert code == correct_code

	def test_decode_single_color(self):
		seq = [x.copy() for x in SingleColor(Config((3, 3), 4, 1, value=127))]
		code = struct.pack('>3I 18B', 3, 3, 4, 127, 127, 127, 127, 127, 127, 127,
		                   127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		frames = [x.copy() for x in self.cls.decoder(code)]
		for i, frame in enumerate(frames):
			np.testing.assert_equal(frame, seq[i])

	def test_decode_moving_edge(self):
		seq = [x.copy() for x in MovingEdge(Config((3, 3), 2, 2))]
		code = struct.pack('>3I 9B BBIB BBIB B BBIB BBIB B BBIB BBIB B', 3, 3, 4, 0,
		                   0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 0, 0, 255, 1, 1, 0, 0,
		                   255, 1, 0, 0, 255, 1, 1, 0, 0, 255, 1, 0, 0, 255, 1, 1,
		                   0, 0)
		frames = [x.copy() for x in self.cls.decoder(code)]
		for i, frame in enumerate(frames):
			np.testing.assert_equal(frame, seq[i])


@invertability_test(caer.CAERLossy, threshold=5)
class TestCAERLossy:
	"""
	Produce sequence with events (x, y, t, v, p):
			(1, 1, 1, 255, +)
			(1, 1, 3, 248, -)
			(0, 0, 4, 255, +)
			(1, 1, 4, 7,   -)
	"""
	def test_encoder(self):
		sequence = simple_seq()
		code = reduce(self.cls.encoder(sequence, threshold=5))
		correct_code = struct.pack('>3I 9B BBIB B B B BBIBBIBBIB B B B B', 3, 3, 6,
		                           0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 4, 0, 0, 0, 0,
		                           255, 1, 1, 248, 0, 3, 7, 0, 4, 0, 0, 0, 0, 0)
		assert code == correct_code

	def test_encoder_with_skips(self):
		sequence = [
		    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 100, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 97, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 96, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 98, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 97, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 94, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 93, 0], [0, 0, 0]], dtype='uint8'),
		    np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype='uint8')
		]
		code = reduce(self.cls.encoder(sequence, threshold=5))
		correct_code = struct.pack(f'>3I 9B BBBB 2BI2BI2BIB BBBB', 3, 3,
		                           len(sequence), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		                           0, 0, 100, 1, 1, 6, 0, 6, 161, 1, 8, 0, 0, 0, 0,
		                           0)
		assert code == correct_code

	def test_encode_moving_edge(self):
		seq = MovingEdge(Config((3, 3), 2, 2))
		code = reduce(self.cls.encoder(seq))
		correct_code = struct.pack('>3I 9B BBIB BBIB B BBIB BBIB B BBIB BBIB B', 3,
		                           3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 0, 0,
		                           255, 1, 1, 0, 0, 255, 1, 0, 0, 255, 1, 1, 0, 0,
		                           255, 1, 0, 0, 255, 1, 1, 0, 0)
		assert code == correct_code

	def test_decode_single_color(self):
		seq = [x.copy() for x in SingleColor(Config((3, 3), 4, 1, value=127))]
		code = struct.pack('>3I 18B', 3, 3, 4, 127, 127, 127, 127, 127, 127, 127,
		                   127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		frames = [x.copy() for x in self.cls.decoder(code)]
		for i, frame in enumerate(frames):
			np.testing.assert_equal(frame, seq[i])

	def test_decode_moving_edge(self):
		seq = [x.copy() for x in MovingEdge(Config((3, 3), 2, 2))]
		code = struct.pack('>3I 9B BBIB BBIB B BBIB BBIB B BBIB BBIB B', 3, 3, 4, 0,
		                   0, 0, 0, 0, 0, 0, 0, 0, 255, 1, 0, 0, 255, 1, 1, 0, 0,
		                   255, 1, 0, 0, 255, 1, 1, 0, 0, 255, 1, 0, 0, 255, 1, 1,
		                   0, 0)
		frames = [x.copy() for x in self.cls.decoder(code)]
		for i, frame in enumerate(frames):
			np.testing.assert_equal(frame, seq[i])
