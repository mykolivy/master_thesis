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

class Config:
    def __init__(self, resolution, fps, duration, dtype='int8'):
        self.width = resolution[0]
        self.height = resolution[1]
        self.fps = fps
        self.duration = duration
        self.frame_num = fps * duration
        self.dtype = dtype

class MovingEdge:
    """Produces consequent frames of moving edge sequence with each iteration"""
    def __init__(self, config):
        self.conf = config
        self.index = 1
        self.frame = np.zeros(self.conf.res, dtype=self.conf.dtype)
        self.step_length = self.conf.width / self.conf.frame_num 
        self.position = 0

    def __iter__(self):
        yield self.frame

        for _ in range(self.conf.frame_num):
            self.position += self.step_length
            self.frame[:, int(self.position)] = 255
            yield self.frame

class SingleColor:
    """Produces consequent frames of single color sequence with each
    iteration"""
    def __init__(self, value, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.frame = np.full((self.conf.res), value)
        self.start_frame = self.frame

    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        self.index += 1
        yield self.frame

class RandomPixel:
    """Produces consequent frames of random pixel seuquence with each
    iteration"""
    def __init__(self, value_range, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.value_range = value_range
        self.start_frame = self.rand_frame() 
    
    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        self.index += 1
        yield self.rand_frame()

    def rand_frame(self):
        return np.random.randint(self.value_range[0], high=self.value_range[1],
               size=self.conf.res)

class Checkers:
    """Produces consequent frames of checkerboard pattern with each iteration"""
    def __init__(self, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.even = self.frame(even=True)
        self.odd = self.frame(even=False)
        self.start_frame = self.even

    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        result = self.even
        if self.index % 2 != 0:
            result = self.odd
        self.index += 1
        yield result

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
        return mat

class RandomBinaryChange:
    """Produces consequent frames of frames where number of pixel flip their
    values. Number of such pixels depends on event rate specified"""
    def __init__(self, rate, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.start_frame = np.zeros(self.conf.res)
        self.val = self.start_frame.copy()
        self.events = {} 
        self.rate = rate

    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        self.change_rand(self.rate)
        self.index += 1
        yield self.val
   
    def set_from_events(self, events):
        self.events = events.copy()
        for event in events:
            i = int(event / self.conf.res[1])
            j = event - i*self.conf.res[1]
            self.val[i][j] = RandomBinaryChange.invert(self.val[i][j])

    def change_rand(self, rate):
        res_sq = self.conf.res[0] * self.conf.res[1]
        num = int(res_sq * rate)
        population = range(0, res_sq)
        self.set_from_events(random.sample(population, num))

    def invert(val):
        if val == 0:
            return 255
        else: 
            return 0

class RandomChange:
    """Produces consequent frames of frames where number of pixel flip their
    values. Number of such pixels depends on event rate specified"""
    def __init__(self, rate, value_range, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.value_range = value_range
        self.start_frame = np.zeros(self.conf.res)
        self.val = self.start_frame.copy()
        self.events = {} 
        self.rate = rate

    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        self.change_rand(self.rate)
        self.index += 1
        yield self.val
   
    def set_from_events(self, events):
        self.events = events.copy()
        for event in events:
            i = int(event / self.conf.res[1])
            j = event - i*self.conf.res[1]
            self.val[i][j] = random.randrange(self.value_range[0], self.value_range[1])
    
    def change_rand(self, rate):
        res_sq = self.conf.res[0] * self.conf.res[1]
        num = int(res_sq * rate)
        population = range(0, res_sq)
        self.set_from_events(random.sample(population, num))

    def invert(val):
        if val == 0:
            return 255
        else: 
            return 0

class RandomChanceChange:
    """Produces sequence of frames. In each frame each pixel has rate % chance
       to change their value in next frame"""
    def __init__(self, rate, value_range, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.value_range = value_range
        self.start_frame = np.zeros(self.conf.res)
        self.frame = np.zeros(self.conf.res)
        self.next = self.frame
        self.rate = rate

    def __iter__(self):
        if self.index == self.conf.frame_num:
            return
        self.frame = self.next.copy()
        for i, row in enumerate(self.next):
            for j, x in enumerate(row):
                if random.uniform(0, 1) <= self.rate:
                    self.next[i][j] = random.randrange(self.value_range[0], 
                                                      self.value_range[1])
        self.index += 1
        yield self.frame
