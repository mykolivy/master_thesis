import sys, random
import numpy as np

input_files = sys.argv[1:-1]
output_file = sys.argv[-1]
print(f"Extracting from {input_files} into {output_file}")

EVENT_NUM = 128


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


def polarity_map(p):
	return "-1.0" if p == 0 else "1.0"


def events_to_line(events):
	ts = "0.0 " + " ".join(map(str, np.diff(events["t"])))
	ps = " ".join(map(polarity_map, events["p"]))
	return f"{ts} {ps}\n"


total_events = 0
with open(output_file, "w") as fout:
	for path in input_files:
		print(f"Extracting events from {path}...")
		with open(path, "r") as fin:
			width, height = [int(x) for x in fin.readline().split(" ")]
			matrix = [[{
			    "t": [],
			    "p": []
			} for y in range(height)] for x in range(width)]
			lines = []

			for line in fin:
				t, x, y, p = to_event(line)
				matrix[x][y]["t"].append(t)
				matrix[x][y]["p"].append(p)
				if len(matrix[x][y]["t"]) == EVENT_NUM:
					lines.append(events_to_line(matrix[x][y]))
					matrix[x][y] = {"t": [], "p": []}

				if len(lines) >= 10000:
					fout.writelines(lines)
					total_events += len(lines)
					lines = []

			fout.writelines(lines)
			total_events += len(lines)

print(f"Total events extracted: {total_events}")

# Loading dataset
# import tensorflow as tf
# record_defaults = [float()] * 6
# filenames = ["/home/dumpling/Documents/uni/thesis/test.txt"]
# # dataset = tf.data.experimental.CsvDataset(filenames,
# #                                           record_defaults=record_defaults,
# #                                           field_delim=' ').batch(4)
# # for x in dataset:
# # 	print(tf.reshape(tf.transpose(tf.stack(x)), (4, 2, -1)))
# # 	break

# def string_to_tensor(x):
# 	return tf.reshape(tf.strings.to_number(tf.strings.split(x, sep=' ')), (2, -1))

# print("ANOTHER")
# dataset = tf.data.TextLineDataset(
#     filenames, compression_type=None, buffer_size=None,
#     num_parallel_reads=None).map(string_to_tensor).shuffle(10000).batch(2)
# for x in dataset:
# 	print(x)
# 	break
