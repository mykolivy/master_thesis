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

	def __iter__(self) -> np.ndarray:
		cap = cv2.VideoCapture(self.src.name)
		success, self.frame = cap.read()
		while success:
			yield self.frame
			success, self.frame = cap.read()

	def __len__(self):
		return 0


class FileBytes:
	def __init__(self, src):
		self.src = src

	def __iter__(self):
		with open(self.src, 'rb') as f:
			yield f.read()
