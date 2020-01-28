import numpy as np

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

class AERIterator:
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

class CAERIterator:
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
                     

