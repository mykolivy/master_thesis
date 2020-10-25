import sys, os, glob, imageio
import numpy as np

left_directory = sys.argv[1]
right_directory = sys.argv[2]


def get_frames(directory):
	result = {}
	for filename in sorted(glob.glob(f"{directory}/*.png")):
		result[filename.split('/')[-1]] = imageio.imread(filename)
	return result


left_frames = get_frames(left_directory)
right_frames = get_frames(right_directory)

breakpoint()

assert set(left_frames.keys()) == set(
    right_frames.keys()), "Can not compare two different sequences"

result = {}
for name in left_frames:
	l_frame = left_frames[name]
	r_frame = right_frames[name]
	result[name] = {"mse": np.mean((l_frame - r_frame)**2)}

print(result)
