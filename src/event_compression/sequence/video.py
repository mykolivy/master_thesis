import cv2
import numpy as np
from . import video_sequence


#@video_sequence(name="video")
class VideoSequence:
	"""
	Iterate through frames of regular compressed video files as np.ndarray objects
	"""
	def __init__(self, src):
		self.src = src
		self.frame = None
		self.cap = cv2.VideoCapture(self.src.name)

	def __iter__(self) -> np.ndarray:
		success, self.frame = self.cap.read()
		while success:
			yield self.frame
			success, self.frame = self.cap.read()

	def __len__(self):
		return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


class FileBytes:
	def __init__(self, src):
		self.src = src

	def __iter__(self):
		with open(self.src, 'rb') as f:
			yield f.read()


class GrayscaleVideoConverter:
	def __init__(self, video_seq):
		self.video_seq = video_seq

	def __iter__(self):
		for frame in self.video_seq:
			yield np.round(np.sum(frame, 2) / 3.).astype('uint8')

	def __len__(self):
		return len(self.video_seq)
