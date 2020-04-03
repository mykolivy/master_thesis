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
    def __init__(self, resolution, fps, duration, dtype='int8', rate=None, val_range=None):
        self.width = resolution[0]
        self.height = resolution[1]
        self.res = (self.width, self.height)
        self.fps = fps
        self.duration = duration
        self.frame_num = fps * duration
        self.dtype = dtype
        self.rate = rate
        self.range = val_range

class SingleColor:
    """Produces consequent frames of single color sequence with each
    iteration"""
    def __init__(self, value, config):
        self.conf = config
        self.frame = np.full((self.conf.width, self.conf.height), value)

    def __iter__(self):
        for _ in range(self.conf.frame_num):
            yield self.frame

class MovingEdge:
    """Produces consequent frames of moving edge sequence with each iteration"""
    def __init__(self, config):
        self.conf = config
        self.frame = np.zeros((self.conf.width, self.conf.height), dtype=self.conf.dtype)
        self.step_length = self.conf.width / self.conf.frame_num 
        self.position = 0

    def __iter__(self):
        yield self.frame

        for _ in range(self.conf.frame_num):
            self.position += self.step_length
            self.frame[:, int(self.position)] = 255
            yield self.frame

class RandomPixel:
    """Produces consequent frames of random pixel seuquence with each
    iteration"""
    def __init__(self, value_range, sequence_config):
        self.conf = sequence_config
    
    def __iter__(self):
        for _ in range(self.conf.frame_num):
            yield self.rand_frame()

    def rand_frame(self):
        return np.random.randint(self.conf.range[0], high=self.conf.range[1],
               size=(self.conf.width, self.conf.height))

class Checkers:
    """Produces consequent frames of checkerboard pattern with each iteration"""
    def __init__(self, config):
        self.conf = config
        self.even = self.frame(even=True)
        self.odd = self.frame(even=False)

    def __iter__(self):
        for i in range(self.conf.frame_num):
            if i%2 == 0:
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
        return mat

class RandomBinaryChange:
    """Produces consequent frames where number of pixel flip their
    values. Number of such pixels depends on event rate specified"""
    def __init__(self, config):
        self.conf = config
        self.frame = np.zeros(self.conf.res)
        self.res_sq = self.conf.width * self.conf.height

    def __iter__(self):
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
            j = event - i*self.conf.width
            if self.frame[i,j] == 0:
                self.frame[i,j] = 255
            else:
                self.frame[i,j] = 0

class RandomChange:
    """Produces consequent frames of frames where number of pixel flip their
    values. Number of such pixels depends on event rate specified"""
    def __init__(self, config):
        self.conf = config
        self.frame = np.zeros(self.conf.res)
        self.res_sq = self.conf.widht * self.conf.height

    def __iter__(self):
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
            j = event - i*self.conf.width
            self.frame[i][j] = random.randrange(self.conf.range[0], self.conf.range[1])

class RandomChanceChange:
    """Produces sequence of frames. In each frame each pixel has rate % chance
       to change their value in next frame"""
    def __init__(self, config):
        self.conf = config
        self.frame = np.zeros(self.conf.res)

    def __iter__(self):
        for _ in range(self.conf.frame_num):
            for i, row in enumerate(self.frame):
                for j, _ in enumerate(row):
                    if random.uniform(0, 1) <= self.conf.rate:
                        self.frame[i][j] = random.randrange(self.conf.range[0], 
                                                        self.conf.range[1])
            yield self.frame
