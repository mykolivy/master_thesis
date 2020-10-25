input_file = "data/dynamic_6dof-2.txt"
output_file = "data/time_discrete/dynamic_6dof-2.txt"

#time_window = 0.033
time_window = 1.0


def to_event(line):
	comps = line.split(" ")
	if len(comps) != 4:
		raise Exception("Wrong AER data format")

	t = float(comps[0])
	x, y, p = [int(x) for x in comps[1:]]
	return t, x, y, p


def write_events(events, out_name, width, height):
	with open(f"{out_name}", "w") as out:
		out.write(f"{width} {height}\n")
		events = [f"{e[0]} {e[1]} {e[2]} {e[3]}" for e in events]
		out.write('\n'.join(events))


def append_window_events(time_window_events, events):
	n = len(time_window_events)
	if n != 0:
		t_start = time_window_events[0][0]
		t_end = time_window_events[-1][0]
		delta_t = abs((t_end - t_start) / n)

		for i, e in enumerate(time_window_events):
			t, x, y, p = e
			events.append((t_start + i * delta_t, x, y, p))


with open(input_file, "r") as f:
	width, height = [int(x) for x in f.readline().split(" ")]
	first_event = to_event(f.readline())

	cum_t = 0.
	time_window_events = [first_event]
	events = []
	for line in f:
		event = to_event(line)

		if event[0] <= time_window_events[0][0] + time_window:
			time_window_events.append(event)
		else:
			append_window_events(time_window_events, events)
			time_window_events = [event]

	append_window_events(time_window_events, events)

	write_events(events, output_file, width, height)