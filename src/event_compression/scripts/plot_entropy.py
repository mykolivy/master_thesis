from event_compression.codec import codecs
from event_compression.scripts.threshold_search import compute_entropy
from event_compression.sequence.synthetic import Config, RandomChange
from event_compression.scripts import util
import numpy as np
import os
import functools, operator
import matplotlib.pyplot as plt

args = None


def get_args():
	parser = util.get_parser(__file__)
	return parser.parse_args()


def sequences(x_axis, args):
	for i, x in enumerate(x_axis):
		print(f"Generating sequence {i}, rate = {x:1.3}...")
		seq_conf = Config(args.res, 1, args.duration, rate=x)
		seq = RandomChange(seq_conf)
		yield seq


def main():
	args = get_args()

	x = np.linspace(*args.range, args.samples)
	codec_names = ["raw", "aer", "residual"]
	codec_colors = ["k", "b", "r"]
	results = {name: [] for name in codec_names}

	for seq in sequences(x, args):
		for codec_name in codec_names:
			print(f"{codec_name}: ", end="", flush=True)
			codec = codecs()[codec_name]
			bytes_seq = functools.reduce(operator.add, codec.encoder(seq),
			                             bytearray())
			result = compute_entropy([bytes_seq]) * len(bytes_seq)
			results[codec_name].append(result)
			print(result)
	print("SUMMARY:")
	print(f"x: {x}")
	print(f"y: {results}")
	print("")

	print("Plotting graphs...")
	for codec_name, color in zip(codec_names, codec_colors):
		plt.plot(x, results[codec_name], color, label=codec_name)

	plt.xlabel('Event Rate')
	plt.ylabel('Entropy * Signal length')
	plt.title(
	    f"Entropy Lengths res:{args.res}, dur:{args.duration}, samples:{args.samples}"
	)
	plt.legend()

	plt.show()


if __name__ == "__main__":
	main()