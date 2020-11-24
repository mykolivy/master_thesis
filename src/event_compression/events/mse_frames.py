import sys, os, glob, imageio
import numpy as np
import matplotlib.pyplot as plt

left_directory = sys.argv[1]
right_directory = sys.argv[2]


def get_frames(directory):
	files = sorted(glob.glob(f"{directory}/*.png"))
	for x in files:
		yield imageio.imread(x)
	# result = [imageio.imread(x) for x in files]
	# result[filename.split('/')[-1]] = imageio.imread(filename)
	# return result


left_frames = get_frames(left_directory)
right_frames = get_frames(right_directory)

# breakpoint()
# assert len(left_frames) == len(right_frames)

result = []

for l_frame, r_frame in zip(left_frames, right_frames):
	result.append({"mse": np.mean((l_frame - r_frame)**2)})

# print(result)

# Calculate mean
mse = [x['mse'] for x in result]
print(f"Mean MSE: {np.mean(mse)}")

# Plot
# plt.plot(mse)
# plt.ylabel("MSE")
# plt.xlabel("Frame")
# plt.title(f"{left_directory} vs {right_directory}")
# plt.show()