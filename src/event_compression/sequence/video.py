import cv2
import numpy as np
from . import video_sequence


#@video_sequence(name="video")
class VideoSequence:
	"""
	Iterate through frames of regular compressed video files as np.ndarray objects
	"""
	def __init__(self, config):
		self.conf = config
		self.src = config.video
		self.frame = None

	def __iter__(self) -> np.ndarray:
		cap = cv2.VideoCapture(self.src)
		success, self.frame = cap.read()
		while success:
			yield self.frame
			success, self.frame = cap.read()

	def __len__(self):
		return self.conf.frame_num