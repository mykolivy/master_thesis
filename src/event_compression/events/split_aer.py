"""
Splits `input_file` aer sequence into n shorter consecutive subsequences.
Each subsequence's duration is determined by `time_interval`.
"""

input_file = "dynamic_6dof.txt"
time_interval = 1.0
out_name = "dynamic_6dof"


def to_event(line):
	comps = line.split(" ")
	if len(comps) != 4:
		raise Exception("Wrong AER data format")

	t = float(comps[0])
	x, y, p = [int(x) for x in comps[1:]]
	return t, x, y, p


def write_events(events, out_name, index, width, height):
	with open(f"{out_name}-{index}.txt", "w") as out:
		out.write(f"{width} {height}\n")
		events = [f"{e[0]} {e[1]} {e[2]} {e[3]}" for e in events]
		out.write('\n'.join(events))


index = 0
with open(input_file, "r") as f:
	width, height = [int(x) for x in f.readline().split(" ")]
	first_event = to_event(f.readline())

	cum_t = 0.
	prev_t = first_event[0]
	events = []
	for line in f:
		event = to_event(line)

		delta_t = event[0] - prev_t
		cum_t += event[0] - prev_t
		prev_t = event[0]

		events.append(event)

		if cum_t >= time_interval:
			write_events(events, out_name, index, width, height)
			index += 1
			cum_t = 0.0
			events = []
			print("Created subsequence {index}.")

	if len(events) != 0:
		write_events(events, out_name, index, width, height)
