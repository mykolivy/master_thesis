import cv2
import numpy as np
from . import _REGISTER

def codecs():
	return _REGISTER.copy()

def codec(name=None):
	def decorate(cls):
		_REGISTER[name] = cls
		return cls
	return decorate

class VideoFrameReader:
    """Iterate through frames of regular compressed video files as np.ndarray objects"""
    def __init__(self, video):
        self.src = video
        self.frame = None

    def __iter__(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.src)
        success, self.frame = cap.read()
        while success:
            yield self.frame
            success, self.frame = cap.read()

class EventCodec:
    """Read/write file in specific event codec"""
    def __init__(self, format, resolution, duration, fps=30):
        self.format = format
        self.res = resolution
        self.duration = duration
        self.fps = fps
       
    @classmethod
    def write(content, file, codec=None):
        file.write(self.format)
        file.write(self.res)
        file.write(self.duration)
        file.write(self.fps)

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