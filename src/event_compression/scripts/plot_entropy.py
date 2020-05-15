from event_compression.codec import codecs
from event_compression.scripts.threshold_search import compute_entropy
from event_compression.sequence.synthetic import Config, RandomChange
import numpy as np
import os
import functools, operator
import matplotlib.pyplot as plt

res = 32
duration = 30
start = 0.
end = 1 / (32 * 32)
samples = 100

raw = codecs()["raw"]
aer = codecs()["aer"]
caer = codecs()["caer"]
residual = codecs()["residual"]


def entropy(codec):
	def f(x):
		seq_conf = Config((res, res), 1, duration, rate=x)
		seq = RandomChange(seq_conf)
		return compute_entropy(codec.encoder(seq))

	return f


def entropy_size(codec):
	def f(x):
		seq_conf = Config((res, res), 1, duration, rate=x)
		seq = RandomChange(seq_conf)
		bytes_seq = functools.reduce(operator.add, codec.encoder(seq), bytearray())
		return compute_entropy([bytes_seq]) * len(bytes_seq)

	return f


def plot_entropies():
	x = np.linspace(start, end, samples)

	y = np.vectorize(entropy(raw))(x)
	plt.plot(x, y, "k", label='RAW')

	y = np.vectorize(entropy(aer))(x)
	plt.plot(x, y, "b", label='AER')

	y = np.vectorize(entropy(caer))(x)
	plt.plot(x, y, "gold", label='CAER')

	y = np.vectorize(entropy(residual))(x)
	plt.plot(x, y, "r", label='Residual')

	plt.xlabel('Event Rate')
	plt.ylabel('Entropy')
	plt.title("Entropies")
	plt.legend()

	plt.show()


def plot_entropy_sizes():
	x = np.linspace(start, end, samples)

	y = np.vectorize(entropy_size(raw))(x)
	plt.plot(x, y, "k", label='RAW')

	y = np.vectorize(entropy_size(aer))(x)
	plt.plot(x, y, "b", label='AER')

	y = np.vectorize(entropy_size(caer))(x)
	plt.plot(x, y, "gold", label='CAER')

	y = np.vectorize(entropy_size(residual))(x)
	plt.plot(x, y, "r", label='Residual')

	plt.xlabel('Event Rate')
	plt.ylabel('Entropy * Signal length')
	plt.title("Entropy Lengths")
	plt.legend()

	plt.show()


plot_entropies()
plot_entropy_sizes()