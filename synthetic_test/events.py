import numpy as np
import struct

def create_raw_file(f, resolution, fps, duration):
    f.write((resolution[0]).to_bytes(4, byteorder='big'))
    f.write((resolution[1]).to_bytes(4, byteorder='big'))
    f.write((fps).to_bytes(1, byteorder='big'))
    f.write((duration).to_bytes(4, byteorder='big'))

def save_frame(out, data):
    for row in np.uint8(data):
        for j in row:
            out.write(int(j).to_bytes(1, byteorder='big'))
        
def save_event_frame(diff, t, out):
    for i, row in enumerate(diff):
        for j, value in enumerate(row):
            if value != 0:
                out.write(i.to_bytes(4, byteorder='big', signed=False))
                out.write(j.to_bytes(4, byteorder='big', signed=False))
                out.write(t.to_bytes(4, byteorder='little', signed=False))
                if value >= 0:
                    out.write(int(1).to_bytes(1, byteorder='big', signed=False))
                else:
                    out.write(int(2).to_bytes(1, byteorder='big', signed=False))

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

# Each event is represented as: 
#   i (4 bytes), j (4 bytes), t (float), value (1 byte), polarity (1 byte)
# If value >= 127, two events are created at the same timestamp (127 and
# residual), so that each value can be stored with one byte
class AERIterator:
    def __init__(self, frame_iterator):
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame
        self.overflow = bytearray()

    def __iter__(self):
        return self

    def __next__(self):
        for t, frame in enumerate(self.frames):
            diff = np.subtract(frame, self.prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    value = int(value)
                    if value == 0:
                        continue
                    result += i.to_bytes(4, byteorder='big', signed=False)
                    result += j.to_bytes(4, byteorder='big', signed=False)
                    result += bytearray(struct.pack("f", t+1))

                    sign = 0 if value >= 0 else 1
                    if abs(value) >= 127:
                        result += int(127).to_bytes(1, byteorder='big', signed=False)
                        overflow = int(abs(value) - 127)
                        result += i.to_bytes(4, byteorder='big', signed=False)
                        result += j.to_bytes(4, byteorder='big', signed=False)
                        #shift timestamp by a half
                        #result += bytearray(struct.pack("f", t+1.5))
                        #use the same timestamp
                        result += bytearray(struct.pack("f", t+1))
                        result += int(sign & overflow).to_bytes(1,
                                  byteorder='big', signed=False)
                    else:
                        result += int(sign & abs(value)).to_bytes(1,
                                byteorder='big', signed=False)
            self.prev = frame.copy()
            yield result

class AERByteBinaryIterator:
    def __init__(self, frame_iterator):
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame

    def __iter__(self):
        return self

    def __next__(self):
        for t, frame in enumerate(self.frames):
            diff = np.subtract(frame, self.prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    if value == 0:
                        continue
                    result += i.to_bytes(4, byteorder='big', signed=False)
                    result += j.to_bytes(4, byteorder='big', signed=False)
                    result += (t+1).to_bytes(4, byteorder='little', signed=False)
                    if value >= 0:
                        result += int(1).to_bytes(1, byteorder='big', signed=False)
                    else:
                        result += int(2).to_bytes(1, byteorder='big', signed=False)
            self.prev = frame.copy()
            yield result

# Same as AERIterator, but lossy.
# Events with value smaller or equal to the threshold are ignored.
# Consecutive ignored values are accumulated, until their sum is > threshold, in
# which case a new event with value of this sum is created at the current
# timestamp.
class AERLossyIterator:
    def __init__(self, frame_iterator, threshold=8):
        self.threshold = threshold
        self.res = frame_iterator.conf.res
        self.ignored_sums = np.zeros((self.res[0], self.res[1]))
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame
        self.overflow = bytearray()

    def __iter__(self):
        return self

    def __next__(self):
        for t, frame in enumerate(self.frames):
            diff = np.subtract(frame, self.prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    if value == 0:
                        continue
                    elif abs(value) <= self.threshold:
                        self.ignored_sums[i][j] += value
                        if abs(self.ignored_sums[i][j]) > self.threshold:
                            AERLossyIterator.event_to_bytes(i,j,t+1,
                                    self.ignored_sums[i][j], result)
                            self.ignored_sums[i][j] = 0
                    else:
                        self.ignored_sums[i][j] = 0
                        AERLossyIterator.event_to_bytes(i,j,t+1,value,result)
            self.prev = frame.copy()
            yield result

    def event_to_bytes(i, j, t, value, result):
        value = int(value)
        result += i.to_bytes(4, byteorder='big', signed=False)
        result += j.to_bytes(4, byteorder='big', signed=False)
        result += bytearray(struct.pack("f", t))

        sign = 0 if value >= 0 else 1
        if abs(value) >= 127:
            result += int(127).to_bytes(1, byteorder='big', signed=False)
            overflow = int(abs(value) - 127)
            result += i.to_bytes(4, byteorder='big', signed=False)
            result += j.to_bytes(4, byteorder='big', signed=False)
            #use the same timestamp
            result += bytearray(struct.pack("f", t))
            result += int(sign & overflow).to_bytes(1,
                      byteorder='big', signed=False)
        else:
            polarity = int(sign & abs(value))
            result += polarity.to_bytes(1, byteorder='big', signed=False)


class CAERBinaryDeltaIterator:
    def __init__(self, frame_iterator):
        self.frames = frame_iterator
        self.prev = frame_iterator.start_frame
        self.res = self.frames.conf.res
        self.arranged = None

    def __iter__(self):
        return self

    def __next__(self):
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
        return self

    def __next__(self):
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


format_iterators = {
        'aer': AERByteBinaryIterator,
        'aer_true': AERIterator,
        'aer_lossy': AERLossyIterator,
        'caer': CAERBinaryDeltaIterator,
        'caer_true': CAERIterator
}
