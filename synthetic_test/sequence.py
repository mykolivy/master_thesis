import numpy as np

class SequenceConfig:
    def __init__(self, resolution, fps, duration):
        self.res = resolution
        self.fps = fps
        self.duration = duration
        self.frame_num = fps * duration

class MovingEdgeFrameIterator:
    """Produces consequent frames of moving edge sequence with each iteration"""
    def __init__(self, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.data = np.zeros((self.conf.res, self.conf.res))
        self.start_frame = self.data.copy()
        self.move_ratio = int(self.conf.frame_num / self.conf.res)
        self.row = np.zeros(self.conf.res)
        self.col_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.conf.frame_num:
            raise StopIteration
        if self.index % self.move_ratio == 0:
            for i in range(0,self.conf.res):
                self.data[i][self.col_index] = 255
            self.col_index=(self.col_index+1)%self.conf.res
        self.index += 1
        return self.data

class SingleColorFrameIterator:
    """Produces consequent frames of single color sequence with each
    iteration"""
    def __init__(self, value, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.frame = np.full((self.conf.res, self.conf.res), value)
        self.start_frame = self.frame

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.conf.frame_num:
            raise StopIteration
        self.index += 1
        return self.frame

class RandomPixelFrameIterator:
    """Produces consequent frames of random pixel seuquence with each
    iteration"""
    def __init__(self, value_range, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.value_range = value_range
        self.start_frame = self.rand_frame() 
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.conf.frame_num:
            raise StopIteration
        self.index += 1
        return self.rand_frame()

    def rand_frame(self):
        return np.random.randint(self.value_range[0], high=self.value_range[1],
               size=(self.conf.res, self.conf.res))

class CheckersFrameIterator:
    """Produces consequent frames of checkerboard pattern with each iteration"""
    def __init__(self, sequence_config):
        self.conf = sequence_config
        self.index = 1
        self.even = self.frame(even=True)
        self.odd = self.frame(even=False)
        self.start_frame = self.even

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.conf.frame_num:
            raise StopIteration
        result = self.even
        if self.index % 2 != 0:
            result = self.odd
        self.index += 1
        return result

    def frame(self, even):
        mat = np.zeros((self.conf.res, self.conf.res))
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

