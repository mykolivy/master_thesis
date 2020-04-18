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


@video_sequence(name="random_binary_change")
class RandomBinaryChange:
	"""Produces consequent frames where number of pixel flip their
    values. Number of such pixels depends on event rate specified"""
	def __init__(self, config):
		self.conf = config
		self.frame = np.zeros(self.conf.res, dtype=self.conf.dtype)
		self.res_sq = self.conf.width * self.conf.height
		self.seed = get_seed()

	def __iter__(self):
		random.seed(self.seed)
		for _ in range(self.conf.frame_num):
			self.change_rand()
			yield self.frame

	def change_rand(self):
		num = int(self.res_sq * self.conf.rate)
		population = range(0, self.res_sq)
		self.set_from_events(random.sample(population, num))

	def set_from_events(self, events):
		for event in events:
			i = int(event / self.conf.width)
			j = event - i * self.conf.width
			if self.frame[i, j] == 0:
				self.frame[i, j] = 255
			else:
				self.frame[i, j] = 0

	def __len__(self):
		return self.conf.frame_num


@video_sequence(name="random_change")
class RandomChange:
	"""Produces consequent frames of frames where number of pixel flip their
    values. Number of such pixels approximates event rate specified"""
	def __init__(self, config):
		self.conf = config
		self.res_sq = self.conf.width * self.conf.height
		self.seed = get_seed()
		self.first_frame = self.compute_first_frame()

	def __iter__(self):
		random.seed(self.seed)
		frame = self.first_frame.copy()
		for _ in range(self.conf.frame_num):
			num = int(self.res_sq * self.conf.rate)
			population = range(0, self.res_sq)
			self.set_from_events(frame, random.sample(population, num))
			yield frame

	def set_from_events(self, frame, events):
		for event in events:
			i = int(event / self.conf.width)
			j = event - i * self.conf.width
			frame[i][j] = random.randrange(self.conf.range[0], self.conf.range[1])

	def __len__(self):
		return self.conf.frame_num

	def compute_first_frame(self):
		frame = np.empty(self.conf.res, dtype=self.conf.dtype)
		for index, _ in np.ndenumerate(frame):
			frame[index] = random.randrange(self.conf.range[0], self.conf.range[1])
		return frame


@video_sequence(name="random_chance_change")
class RandomChanceChange:
	"""Produces sequence of frames. In each frame each pixel has rate % chance
       to change their value in next frame"""
	def __init__(self, config):
		self.conf = config
		self.seed = get_seed()

		self.first_frame = self.compute_first_frame()

	def __iter__(self):
		random.seed(self.seed)

		frame = self.first_frame.copy()
		for _ in range(self.conf.frame_num):
			frame = self.change(frame)
			yield frame

	def __len__(self):
		return self.conf.frame_num

	def change(self, frame):
		for index, _ in np.ndenumerate(frame):
			if random.random() <= self.conf.rate:
				frame[index] = random.randrange(self.conf.range[0], self.conf.range[1])
		return frame

	def compute_first_frame(self):
		frame = np.empty(self.conf.res, dtype=self.conf.dtype)
		for index, _ in np.ndenumerate(frame):
			frame[index] = random.randrange(self.conf.range[0], self.conf.range[1])
		return frame


@video_sequence(name="random_adaptive_change")
class RandomAdaptiveChange:
	"""Produces consequent frames of frames where number of pixel flip their
    values. Number of such pixels approximates event rate specified"""
	def __init__(self, config):
		self.conf = config
		self.first_frame = np.zeros(self.conf.res, dtype=self.conf.dtype)
		self.res_sq = self.conf.res[0] * self.conf.res[1]
		self.seed = get_seed()
		self.changable_num = int(self.res_sq * self.conf.rate)

	def __iter__(self):
		random.seed(self.seed)

		total_changed = 0
		adapt = 0
		frame = self.first_frame.copy()
		for i in range(self.conf.frame_num):
			num = self.changable_num + adapt
			population = range(0, self.res_sq)
			self.set_from_events(frame, random.sample(population, num))

			total_changed += num
			adapt = self.get_adaptation_num(i + 1, total_changed)

			yield frame

	def set_from_events(self, frame, events):
		for event in events:
			i = int(event / self.conf.res[1])
			j = event - i * self.conf.res[1]
			frame[i][j] = random.randrange(self.conf.range[0], self.conf.range[1])

	def __len__(self):
		return self.conf.frame_num

	def compute_current_rate(self, total_changed):
		return total_changed / (self.res_sq * self.conf.frame_num)

	def get_adaptation_num(self, i, total_changed):
		adapt = round(self.conf.rate * self.res_sq * i) - total_changed
		if adapt < 0:
			return 0
		return adapt * 2
