import pytest
import numpy as np
import event_compression.synthetic.sequence as sequence
import event_compression.synthetic as synthetic
from copy import deepcopy
from decimal import Decimal
import event_compression.analysis.event as analysis


@pytest.fixture
def seq_conf():
	return sequence.Config((3, 3), 4, 1, value=255, rate=0.5, val_range=(0, 255))


def common_test(cls=None):
	"""
	Define common tests for all sequences.
    
	@common_test(cls=sequence_class_to_be_tested)
	class TestSequenceClass:
		pass
	"""
	def decorate(_cls):
		_cls.cls = cls

		def test_resolution(instance, seq_conf):
			seq = instance.cls(seq_conf)
			assert seq.conf.res == seq_conf.res
			for frame in seq:
				assert frame.shape == seq.conf.res

		_cls.test_resolution = test_resolution

		def test_type(instance, seq_conf):
			seq = instance.cls(seq_conf)
			for frame in seq:
				assert frame.dtype == seq_conf.dtype

		_cls.test_type = test_type

		def test_frame_number(instance, seq_conf):
			seq = instance.cls(seq_conf)
			frame_num = len(list(seq))
			correct_num = seq_conf.fps * seq_conf.duration
			assert frame_num == correct_num

		_cls.test_frame_number = test_frame_number

		def test_multiple_iterators_same(instance, seq_conf):
			seq = instance.cls(seq_conf)
			it1 = iter(seq)
			it2 = iter(seq)
			frames1 = [x.copy() for x in it1]
			frames2 = [x.copy() for x in it2]

			assert all([np.all(np.equal(a, b)) for a, b in zip(frames1, frames2)])

		_cls.test_multiple_iterators_same = test_multiple_iterators_same

		return _cls

	return decorate


def value_range_test():
	"""
	Test range of values in the sequence.

	# Arguments
		range: range of values that all the values in the sequence have to lie in.
			range[0] -- inclusive beginning of range.
			range[1] -- exclusive end of range.
	"""
	def decorate(cls):
		ranges = [(0, 256), (1, 3)]

		def f(cls, range, seq_conf):
			seq_conf.range = range
			frames = [x.copy() for x in cls.cls(seq_conf)]
			for frame in frames:
				for _, value in np.ndenumerate(frame):
					assert range[0] <= value and value < range[1]

		for range in ranges:

			def caller(cls, range):
				return lambda instance, seq_conf: f(cls, range, seq_conf)

			setattr(cls, f'test_value_range_{range[0]}_{range[1]}',
			        caller(cls, range))

		return cls

	return decorate


def change_rate_test():
	"""
	Test rate of change of a sequence.
	
	# Arguments
		rate: 0 <= int <= 1, Target rate of change of pixel values.
	"""
	def decorate(cls):
		rates = [0., 1., 0.5, 0.1, 0.7]

		def f(cls, rate, seq_conf):
			seq_conf.rate = rate
			res = rate * 100
			res = 32 if res == 0 else res
			seq_conf.res = (res, res)
			frames = [x.copy() for x in cls.cls(seq_conf)]

			prev = frames[0]
			res = prev.shape[0]
			target_changed_num = rate * res * res
			for frame in frames[1:]:
				diff = np.subtract(frame, prev)
				assert np.count_nonzero(diff) == target_changed_num
				prev = frame.copy()

		for rate in rates:

			def caller(cls, rate):
				return lambda instance, seq_conf: f(cls, rate, seq_conf)

			setattr(cls, f'test_rate_change_{rate}', caller(cls, rate))

		return cls

	return decorate


def test_sequence_collection():
	assert synthetic.sequences()


@common_test(cls=sequence.SingleColor)
class TestSingleColor:
	def test_value_of_all_frames(self, seq_conf):
		seq = sequence.SingleColor(seq_conf)
		correct_frame = np.full(seq_conf.res, seq_conf.value, dtype=seq_conf.dtype)
		for i, frame in enumerate(seq):
			print(i)
			np.testing.assert_equal(frame, correct_frame)


@common_test(cls=sequence.MovingEdge)
class TestMovingEdge:
	def test_first_frame_blank(self, seq_conf):
		seq = sequence.MovingEdge(seq_conf)
		correct_frame = np.zeros(seq_conf.res)
		for frame in seq:
			np.testing.assert_equal(frame, correct_frame)
			break


@common_test(cls=sequence.RandomPixel)
class TestRandomPixel:
	pass


@common_test(cls=sequence.Checkers)
class TestCheckers:
	pass


@common_test(cls=sequence.RandomBinaryChange)
class TestRandomBinaryChange:
	pass


@common_test(cls=sequence.RandomChange)
class TestRandomChange:
	pass


@value_range_test()
@common_test(cls=sequence.RandomChanceChange)
class TestRandomChanceChange:
	pass


@common_test(cls=sequence.RandomAdaptiveChange)
class TestRandomAdaptiveChange:
	def test_average_rate_change(self, seq_conf):
		target_rate = 0.4235971
		seq_conf = sequence.Config((32, 32), 500, 1, rate=target_rate)

		seq = self.cls(seq_conf)
		rate = analysis.event_rate(seq)

		assert target_rate == pytest.approx(rate, 0.001)