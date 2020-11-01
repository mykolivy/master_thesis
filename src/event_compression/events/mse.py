import sys, random
import numpy as np

scale = (1., 1.)

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


def time_window_to_matrix(time_window, width, height):
	matrix = [[[] for y in range(height)] for x in range(width)]
	for event in time_window:
		matrix[event[1]][event[2]].append(event)
	return matrix


def append_window_events(time_window_events, events, width, height):
	matrix = time_window_to_matrix(time_window_events, width, height)

	perturbed_window = []
	for row in matrix:
		for timeline in row:
			if len(timeline) == 0:
				continue
			elif len(timeline) == 1:
				perturbed_window.extend(timeline)
			else:
				t_start = timeline[0][0]
				t_end = timeline[-1][0]
				delta_t = abs((t_end - t_start) / (len(timeline) - 1))

				perturbed_timeline = [(t_start + i * delta_t, ) + x[1:]
				                      for i, x in enumerate(timeline)]

				perturbed_window.extend(perturbed_timeline)

	time_window_events = sorted(perturbed_window, key=lambda x: x[0])

	events.extend(time_window_events)


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
				left = np.array(matl[i][j][:min_len])
				right = np.array(matr[i][j][:min_len])

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
	result = np.zeros(left_dims + [2])
	n = 0
	counter = 0
	for x, y in zip(iterate_events(lfin), iterate_events(rfin)):
		# Accumulate
		matl[x[1]][x[2]].append((x[0], x[3]))
		matr[y[1]][y[2]].append((y[0], y[3]))
		n += 1
		counter += 1

		# Compute and clean
		if counter >= 10000:
			compute_accumulated(width, height, matl, matr, result)
			counter = 0
	compute_accumulated(width, height, matl, matr, result)
	breakpoint()
	# Add unmatched
	result = np.mean(np.sqrt(result))
	print(f"EVENT MSE: {result}")