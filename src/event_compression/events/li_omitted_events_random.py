import sys, random

input_file = sys.argv[1]
output_file = sys.argv[2]

fraction = 0.01
total_discarded = 0
total_events = 0
if len(sys.argv) == 4:
	fraction = float(sys.argv[3])


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


with open(input_file, "r") as f:
	width, height = [int(x) for x in f.readline().split(" ")]
	with open(f"{output_file}", "w") as out:
		out.write(f"{width} {height}\n")
		first_event = to_event(f.readline())

		num = 1
		events = []
		for line in f:
			num += 1
			event = to_event(line)

			if random.random() > fraction:
				events.append(event)
			else:
				total_discarded += 1

			if num % 1000 == 0:
				write_events(events, out)
				events = []

		write_events(events, out)
		print(f"Total events: {num}")

print(f"Total events discarded: {total_discarded}")
