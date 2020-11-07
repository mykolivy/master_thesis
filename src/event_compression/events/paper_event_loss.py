import sys, random
import numpy as np

scale = 1.
left_file = sys.argv[1]
right_file = sys.argv[2]


def to_event(line):
	comps = line.split(" ")
	if len(comps) != 4:
		raise Exception("Wrong AER data format")

	t = float(comps[0])
	x, y, p = [int(x) for x in comps[1:]]
	return t, x, y, p


def write_events(events, out):
	events = [f"{e[0]} {e[1]} {e[2]} {e[3]}" for e in events]
	out.write('\n'.join(events) + '\n')


def read_dims(f):
	return [int(x) for x in f.readline().split(" ")]


def iterate_events(fin):
	for line in fin:
		yield to_event(line)


def compute_accumulated(width, height, matl, matr, result):
	for i in range(width):
		for j in range(height):
			min_len = min(len(matl[i][j]), len(matr[i][j]))
			if min_len != 0:
				left = np.array([x[0] for x in matl[i][j][:min_len]])
				right = np.array([x[0] for x in matr[i][j][:min_len]])
				if left.shape != right.shape:
					breakpoint()

				result[i, j] += np.sum((((left - right) * scale)**2.), axis=0)
				matl[i][j] = matl[i][j][min_len:]
				matr[i][j] = matr[i][j][min_len:]


with open(left_file, "r") as lfin, open(right_file, "r") as rfin:
	left_dims = read_dims(lfin)
	right_dims = read_dims(rfin)
	if left_dims != right_dims:
		raise Exception("Comparing sequences with different dimensions.")
	width, height = left_dims

	matl = [[[] for y in range(height)] for x in range(width)]
	matr = [[[] for y in range(height)] for x in range(width)]
	lframe_acc, rframe_acc = np.zeros(left_dims,
	                                  dtype=np.int64), np.zeros(right_dims,
	                                                            dtype=np.int64)
	result = np.zeros(left_dims)
	counter = 0
	for x, y in zip(iterate_events(lfin), iterate_events(rfin)):
		# Accumulate
		matl[x[1]][x[2]].append((x[0], x[3]))
		matr[y[1]][y[2]].append((y[0], y[3]))
		lframe_acc[x[1], x[2]] += int(x[3])
		rframe_acc[y[1], y[2]] += int(y[3])
		counter += 1

		# Compute and clean
		if counter >= 10000:
			compute_accumulated(width, height, matl, matr, result)
			counter = 0
	compute_accumulated(width, height, matl, matr, result)

	result = np.mean(np.sqrt(result)), np.mean(
	    np.sqrt((lframe_acc - rframe_acc)**2.))

	print(f"EVENT MSE: {result}")
	print(f"Mean MSE: {np.mean(result)}")