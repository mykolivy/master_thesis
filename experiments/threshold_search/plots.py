import matplotlib.pyplot as plt
import numpy as np
from event_compression.codec import codecs
from event_compression.scripts.threshold_search import compute_entropy
from event_compression.sequence.synthetic import Config, RandomChange
from event_compression.scripts import util
import os
import functools, operator


def plot(data, x_label, y_label, title):
	fig, ax = plt.subplots()
	ax.plot(data[x_label], data[y_label])

	ax.set(xlabel=x_label, ylabel=y_label)
	ax.grid()

	fig.savefig(f"{title}.svg")
	plt.show()


# Number of frames: 24
thresholds_by_resolutions = {
    "Resolution": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "Threshold": [
        0.1304347826, 0.1820652174, 0.2085597826, 0.2112771739, 0.203125,
        0.1927118716, 0.1844933551, 0.1766005806, 0.1650113645, 0.1534669296
    ]
}

plot(thresholds_by_resolutions, "Resolution", "Threshold",
     "Thresholds by resolutions")

# Resolution: 64x64
thresholds_by_duration = {
    "Duration": [16, 32, 64, 128, 256, 512, 1024],
    "Threshold": [
        0.1928873698, 0.1925560736, 0.1919332837, 0.1866349348, 0.1812088312,
        0.1742581183, 0.1666146406
    ]
}

plot(thresholds_by_duration, "Duration", "Threshold", "Thresholds by duration")


def sequences(x_axis, resolution, duration):
	for i, x in enumerate(x_axis):
		print(f"Generating sequence {i}, rate = {x:1.3}...")
		seq_conf = Config(resolution, 1, duration, rate=x)
		seq = RandomChange(seq_conf)
		yield seq


RESOLUTION = (32, 32)
DURATION = 30
x = np.linspace(0, 1, 100)
codec_names = ["raw", "short_aer", "residual"]
codec_colors = ["k", "b", "r"]
metrics = ["Entropy", "Size", "Compressed Size"]
results = {metric: {name: [] for name in codec_names} for metric in metrics}

for seq in sequences(x, RESOLUTION, DURATION):
	for codec_name in codec_names:
		print(f"{codec_name}: ", end="", flush=True)
		codec = codecs()[codec_name]
		bytes_seq = functools.reduce(operator.add, codec.encoder(seq), bytearray())
		# Leave one, comment out the others
		size = len(bytes_seq)
		entropy = compute_entropy([bytes_seq])
		entropy_size = size * entropy

		results["Size"][codec_name].append(size)
		results["Entropy"][codec_name].append(entropy)
		results["Compressed Size"][codec_name].append(entropy_size)
		print((entropy, size, entropy_size))

print("SUMMARY:")
print(f"x: {x}")
print(f"y: {results}")
print("")

print("Plotting graphs...")
for metric in metrics:
	for codec_name, color in zip(codec_names, codec_colors):
		plt.plot(x, results[metric][codec_name], color, label=codec_name)

	plt.xlabel('Event Rate')
	plt.ylabel(metric)

	plt.legend()
	plt.savefig(f"{metric}.svg")
	plt.show()
