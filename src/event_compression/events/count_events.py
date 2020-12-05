import sys, random
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

input_files = sys.argv[1:]


def to_event(line):
	comps = line.split(" ")
	if len(comps) != 4:
		raise Exception("Wrong AER data format")

	t = float(comps[0])
	x, y, p = [int(x) for x in comps[1:]]
	return t, x, y, p


quantiles = {
    0.01: [],
    0.05: [],
    0.1: [],
    0.15: [],
    0.2: [],
    0.25: [],
    0.3: [],
    0.4: [],
    0.5: [],
    0.6: [],
    0.7: [],
    0.75: [],
    0.8: [],
    0.9: [],
    0.95: [],
    0.99: []
}
lengths_quantiles = {2**x: [] for x in range(8)}

for fin_path in input_files:
	print(fin_path)
	with open(fin_path, "r") as fin:
		width, height = [int(x) for x in fin.readline().split(" ")]
		counters = np.zeros((width, height))

		for line in fin:
			event = to_event(line)
			counters[event[1], event[2]] += 1

	counters = counters.flatten()
	print(f"Min number: {np.amin(counters)}")
	print(f"Max number: {np.amax(counters)}")
	print(f"Mean number: {np.mean(counters)}")
	print(f"Median number: {np.median(counters)}")
	print(f"Quantiles:")
	for quantile in quantiles:
		result = np.quantile(counters, quantile)
		quantiles[quantile].append(result)
		print(f"{quantile} quantile: {result}")

	print(f"Values and their quantiles:")
	for value in lengths_quantiles:
		quantile = stats.percentileofscore(counters, value)
		lengths_quantiles[value].append(quantile)
		print(f"{value}: {quantile}%")

	# plt.hist(counters)
	# plt.title("Event number histogram")
	# plt.show()
	# print(f"All numbers:\n{counters.tolist()}")

print(f"AVERAGE RESULT:")
print("Quantiles:")
for quantile in quantiles:
	print(
	    f"{quantile} quantile: Mean({np.mean(quantiles[quantile])}), Median({np.median(quantiles[quantile])})"
	)
print("Lengths:")
for length in lengths_quantiles:
	print(
	    f"{length}: Mean({np.mean(lengths_quantiles[length])}, Median({np.median(lengths_quantiles[length])})"
	)
