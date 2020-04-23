"""
Define synthetic video sequences used for testing.

Constraint:
Do not change the frames objects that are returned during iteration.
If needed, copy first.
For better performance, the sequences return and reuse internal representation
of the frames.
"""
import numpy as np
import random
import math
from . import video_sequence
from copy import copy
import decimal
from decimal import Decimal
import time


class Config:
	def __init__(self,
	             resolution,
	             fps,
	             duration,
	             dtype='uint8',
	             value=0,
	             rate=0,
	             val_range=(0, 255)):
		self.width = resolution[0]
		self.height = resolution[1]
		self.res = (self.width, self.height)
		self.fps = fps
		self.duration = duration
		self.frame_num = fps * duration
		self.dtype = dtype
		self.rate = rate
		self.range = val_range
		self.value = value


def get_seed():
	return int(time.time())


@video_sequence(name="single_color")
class SingleColor:
	"""Produces consequent frames of single color sequence with each
    iteration"""
	def __init__(self, config):
		self.conf = config
		self.frame = np.full((self.conf.width, self.conf.height),
		                     self.conf.value,
		                     dtype=self.conf.dtype)

	def __iter__(self):
		for _ in range(self.conf.frame_num):
			yield self.frame

	def __len__(self):
		return self.conf.frame_num


@video_sequence(name="moving_edge")
class MovingEdge:
	"""Produces consequent frames of moving edge sequence with each iteration"""
	def __init__(self, config):
		self.conf = config
		self.first_frame = np.zeros((self.conf.width, self.conf.height),
		                            dtype=self.conf.dtype)
		self.step_length = (self.conf.width - 1) / self.conf.frame_num

	def __iter__(self):
		yield self.first_frame

		frame = self.first_frame.copy()
		position = 0
		for _ in range(self.conf.frame_num - 1):
			position += self.step_length
			frame[:, int(position)] = 255
			yield frame

	def __len__(self):
		return self.conf.frame_num


@video_sequence(name="random_pixel")
class RandomPixel:
	"""
	Produces consequent frames of random pixel sequence with each iteration.
	"""
	def __init__(self, config):
		self.conf = config
		self.seed = get_seed()

	def __iter__(self):
		random.seed(self.seed)
		np.random.seed(self.seed)
		for _ in range(self.conf.frame_num):
			yield self.rand_frame().astype(self.conf.dtype)

	def rand_frame(self):
		return np.random.randint(self.conf.range[0],
		                         high=self.conf.range[1],
		                         size=(self.conf.width, self.conf.height))

	def __len__(self):
		return self.conf.frame_num


@video_sequence(name="checkers")
class Checkers:
	"""Produces consequent frames of checkerboard pattern with each iteration."""
	def __init__(self, config):
		self.conf = config
		self.even = self.frame(even=True)
		self.odd = self.frame(even=False)

	def __iter__(self):
		for i in range(self.conf.frame_num):
			if i % 2 == 0:
				yield self.even
			else:
				yield self.odd

	def frame(self, even):
		mat = np.zeros(self.conf.res)
		if even:
			for row in mat[0::2]:
				row[0::2] = 255
			for row in mat[1::2]:
				row[1::2] = 255
		else:
			for row in mat[0::2]:
				row[1::2] = 255
			for row in mat[1::2]:
				row[0::2] = 255
		return mat.astype(self.conf.dtype)

	def __len__(self):
		return self.conf.frame_num


@video_sequence(name="random_change")
class RandomChange:
	"""
	Sequence of frames, where ``config.rate`` fraction of the pixels change.

	.. warning::
		Given limited resolution and number of frames, it is not possible to 
		produce a sequence with exact change rate.

		Random change produces sequence with rate which is **approximate** 
		(as close as possible) to the given ``conf.rate``.

	Arguments:
		:config: Sequence configuration.
	"""
	def __init__(self, config):
		self.conf = config
		self.first_frame = self.init_first_frame()
		self.res_sq = self.conf.res[0] * self.conf.res[1]
		self.seed = get_seed()
		self.changable_num = int(self.res_sq * self.conf.rate)
		self.dist, self.rate = self.get_change_dist()
		self.range_width = self.conf.range[1] - self.conf.range[0]

	def __iter__(self):
		random.seed(self.seed)

		frame = self.first_frame.copy()
		yield frame

		for change in self.dist:
			population = range(self.res_sq)
			self.set_from_events(frame, random.sample(population, change))
			yield frame

	def get_change_dist(self):
		frame_num = len(self)
		target = self.conf.rate

		total = (frame_num - 1) * self.res_sq

		change = round(target * total / (frame_num - 1))
		dist = np.full(frame_num - 1, change)
		total_changed = change * (frame_num - 1)

		to_change = round(target * total - total_changed)
		if abs(to_change) >= 1:
			self.distribute(dist, to_change)

			total_changed += to_change
		assert round(target * total - total_changed) < 1

		rate = np.sum(dist) / total

		return dist, rate

	def set_from_events(self, frame, events):
		for event in events:
			i = int(event / self.conf.res[1])
			j = event - i * self.conf.res[1]

			new_val = self.change(frame[i, j])
			assert new_val != frame[i, j]
			frame[i, j] = new_val

	def __len__(self):
		return self.conf.frame_num

	def distribute(self, dist, value):
		mod = 1 if value > 0 else -1
		for i in range(abs(value)):
			i = i % len(dist)
			dist[i] += mod
			assert dist[i] <= self.res_sq
		np.random.shuffle(dist)

	def change(self, val):
		mod = random.randrange(self.conf.range[0] + 1, self.conf.range[1])
		return (val + mod) % self.range_width + self.conf.range[0]

	def init_first_frame(self):
		result = np.empty(self.conf.res, dtype=self.conf.dtype)
		for (i, j), _ in np.ndenumerate(result):
			result[i, j] = random.randrange(self.conf.range[0], self.conf.range[1])
		return result
