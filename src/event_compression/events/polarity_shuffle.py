import sys, random

input_file = sys.argv[1]
output_file = sys.argv[2]

time_window = 0.033
if len(sys.argv) == 4:
	time_window = float(sys.argv[3])


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
			polarities = [x[-1] for x in timeline]
			random.shuffle(polarities)
			timeline = [x[:-1] + (polarities[i], ) for i, x in enumerate(timeline)]
			perturbed_window.extend(timeline)

	time_window_events = sorted(perturbed_window, key=lambda x: x[0])

	events.extend(time_window_events)


with open(input_file, "r") as f:
	width, height = [int(x) for x in f.readline().split(" ")]
	with open(f"{output_file}", "w") as out:
		out.write(f"{width} {height}\n")
		first_event = to_event(f.readline())

		cum_t = 0.
		time_window_events = [first_event]
		events = []
		for line in f:
			event = to_event(line)

			if event[0] <= time_window_events[0][0] + time_window:
				time_window_events.append(event)
			else:
				append_window_events(time_window_events, events, width, height)
				write_events(events, out)
				events = []
				time_window_events = [event]

		append_window_events(time_window_events, events, width, height)
		write_events(events, out)