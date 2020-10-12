from pathlib import Path
import math, functools, operator
import numpy as np
from event_compression.codec.aer import AER, PureAER
from event_compression.codec.entropy import Residual
from event_compression.sequence.video import VideoSequence, GrayscaleVideoConverter


def compute_entropy_size(data):
	histogram = np.array([0.0 for x in range(256)])
	size = 0
	for chunk in data:
		for b in chunk:
			histogram[b] += 1
			size += 1
	histogram = histogram / np.sum(histogram)

	entropy = 0.0
	for p in histogram:
		if p > 0:
			entropy -= p * math.log(p, 256)
	return entropy, size


pathlist = sorted(Path(".").glob('*.mp4'))
for path in pathlist:
	print(f"{path}: ", end=" ", flush=True)
	video = GrayscaleVideoConverter(VideoSequence(path))
	res_entropy, res_size = compute_entropy_size(Residual.encoder(video))
	video = GrayscaleVideoConverter(VideoSequence(path))
	aer_entropy, aer_size = compute_entropy_size(AER.encoder(video))
	result = "positive" if aer_entropy * aer_size < res_entropy * res_size else "negative"
	print(f"({result}) {aer_entropy} {aer_size} {res_entropy} {res_size}")
