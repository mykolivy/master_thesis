import pytest
import numpy as np
import pdb
import event_compression.synthetic.sequence as sequence
import event_compression.synthetic as synthetic


@pytest.fixture
def seq_conf():
	return sequence.Config((64, 64),
	                       30,
	                       2,
	                       value=255,
	                       rate=0.1,
	                       val_range=(0, 255))


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

		return _cls

	return decorate


def value_range_test(range=(0, 256)):
	"""
	Test range of values in the sequence.

	# Arguments
		range: range of values that all the values in the sequence have to lie in.
			range[0] -- inclusive beginning of range.
			range[1] -- exclusive end of range.
	"""
	def decorate(cls):
		def test_value_range(instance, seq_conf):
			seq_conf.range = range
			for frame in cls(seq_conf):
				for _, value in np.ndenumerate(frame):
					assert range[0] <= value and value < range[1]

		cls.test_value_range = test_value_range

	return decorate


def change_rate_test(rate=0.1):
	"""
	Test rate of change of a sequence.
	
	# Arguments
		rate: 0 <= int <= 1, Target rate of change of pixel values.
	"""
	def decorate(cls):
		def test_integer_rate(instance, seq_conf):
			for _ in range(1):
				pass
			res = 16 / rate
			seq_conf.rate = rate
			seq_conf.res = (rate * factor, rate * factor)
			frames = [x.copy() for x in cls(seq_conf)]

			prev = frames[0]
			for frame in frames[1:]:
				diff = np.subtract(frame, prev)
				assert len(diff > 0) == target_changed_num
				prev = frame.copy()

		cls.test_integer_rate = test_integer_rate

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


@common_test(cls=sequence.RandomChanceChange)
class TestRandomChanceChange:
	pass
