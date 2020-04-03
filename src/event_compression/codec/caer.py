import numpy as np
import struct

def add_compact_frame(diff, t, arranged):
    for i, row in enumerate(diff):
        for j, value in enumerate(row):
            if value != 0:
                arranged[i][j].append(t)
                if value >= 0:
                    arranged[i][j].append(1)
                else:
                    arranged[i][j].append(2)

def save_compact_frames(arranged, out):
    for row in arranged:
        for x in row:
            for i in range(0,len(x),2):
                out.write(x[i].to_bytes(4, byteorder='little', signed=False))
                out.write(x[i+1].to_bytes(1, byteorder='big', signed=False))
            out.write(int(0).to_bytes(1, byteorder='big', signed=False))

class CAERBinaryDeltaIterator:
    def __init__(self, frame_iterator):
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame
        self.res = self.frames.conf.res
        self.arranged = None

    def __iter__(self):
        if self.arranged == None:
            self.compute_compact_frames()
        for row in self.arranged:
            for x in row:
                result = bytearray()
                for i in range(0,len(x),2):
                    result += x[i].to_bytes(4, byteorder='little', signed=False)
                    result += x[i+1].to_bytes(1, byteorder='big', signed=False)
                result += int(0).to_bytes(1, byteorder='big', signed=False)
                yield result

    def compute_compact_frames(self):
        self.arranged = [[[] for y in range(self.res[1])] for x in range(self.res[0])]
        for t, frame in enumerate(self.frames):
            diff = np.subtract(frame, self.prev)
            self.add_compact_frame(diff, t)
            self.prev = frame.copy()

    def add_compact_frame(self, diff, t):
        for i, row in enumerate(diff):
            for j, value in enumerate(row):
                if value != 0:
                    self.arranged[i][j].append(t)
                    if value >= 0:
                        self.arranged[i][j].append(1)
                    else:
                        self.arranged[i][j].append(2)
                     
class CAERIterator:
    def __init__(self, frame_iterator):
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame
        self.res = self.frames.conf.res
        self.compute_compact_frames()

    def __iter__(self):
        for row in self.arranged:
            for x in row:
                result = bytearray()
                for i in range(0,len(x),2):
                    result += x[i].to_bytes(4, byteorder='little', signed=False)
                    self.append_value(result, x[i+1])
                result += int(0).to_bytes(1, byteorder='big', signed=True)
                yield result

    def compute_compact_frames(self):
        self.arranged = [[[] for y in range(self.res[1])] for x in range(self.res[0])]
        for t, frame in enumerate(self.frames):
            diff = np.subtract(frame, self.prev)
            self.add_compact_frame(diff, t)
            self.prev = frame.copy()

    def add_compact_frame(self, diff, t):
        for i, row in enumerate(diff):
            for j, value in enumerate(row):
                if value != 0:
                    self.arranged[i][j].append(t)
                    self.arranged[i][j].append(value)

    # Converts num to 1-byte or 2-byte signed representation and appends to
    # result. Positive value x is stored as x-1 (0 counts as +1)
    # num != 0, -128 < num < 127
    def append_value(self, result, num):
        if abs(num) >= 127:
            sign = 1 if num > 0 else -1   
            result += int(sign*126).to_bytes(1, byteorder='big', signed=True)
            result += int(num-sign*128).to_bytes(1, byteorder='big', signed=True)
        else:
            print('single byte')
            result += num.to_bytes(1, byteorder='big', signed=True)