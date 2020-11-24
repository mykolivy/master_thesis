"""
Takes a scv file (containing floats) as an input.
Produces
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

# data_csv = sys.argv[1]
# original_sample = [float(x) for x in open(data_csv, "r")]

# Samples for AER vs RAW (res: 32x32, frames: 10, range: (0,100), precision: 0.0001)
# original_sample = [
#     0.2017144097222222, 0.2017144097222222, 0.2016059027777778,
#     0.2016059027777778, 0.2016059027777778, 0.2017144097222222, 0.2021484375,
#     0.2013888888888889, 0.2016059027777778, 0.2016059027777778,
#     0.2016059027777778, 0.2016059027777778, 0.2021484375, 0.2019314236111111,
#     0.2021484375, 0.201171875, 0.2017144097222222, 0.2009548611111111,
#     0.2013888888888889, 0.201171875, 0.2016059027777778, 0.2016059027777778,
#     0.2009548611111111, 0.2019314236111111, 0.2009548611111111,
#     0.2016059027777778, 0.2016059027777778, 0.2013888888888889, 0.2021484375,
#     0.201171875, 0.2019314236111111, 0.2021484375, 0.2013888888888889,
#     0.2019314236111111, 0.2013888888888889, 0.2016059027777778,
#     0.2016059027777778, 0.2016059027777778, 0.2013888888888889,
#     0.2016059027777778, 0.201171875, 0.2016059027777778, 0.2016059027777778,
#     0.2016059027777778, 0.2013888888888889, 0.2016059027777778,
#     0.2023654513888889, 0.2013888888888889, 0.2019314236111111, 0.201171875,
#     0.2013888888888889, 0.2013888888888889, 0.2019314236111111,
#     0.2016059027777778, 0.2016059027777778, 0.2016059027777778,
#     0.2009548611111111, 0.2013888888888889, 0.2023654513888889,
#     0.2017144097222222, 0.2007378472222222, 0.2019314236111111,
#     0.2013888888888889, 0.2016059027777778, 0.2016059027777778,
#     0.2013888888888889, 0.2016059027777778, 0.2017144097222222,
#     0.2016059027777778, 0.201171875, 0.2017144097222222, 0.2016059027777778,
#     0.2016059027777778, 0.2019314236111111, 0.2009548611111111,
#     0.2013888888888889, 0.2019314236111111, 0.2023654513888889,
#     0.2009548611111111, 0.2017144097222222, 0.2013888888888889,
#     0.2019314236111111, 0.2016059027777778, 0.2019314236111111,
#     0.2019314236111111, 0.2013888888888889, 0.2013888888888889,
#     0.2016059027777778, 0.2007378472222222, 0.2013888888888889,
#     0.2013888888888889, 0.2009548611111111, 0.2019314236111111,
#     0.2019314236111111, 0.201171875, 0.2009548611111111, 0.2021484375,
#     0.2016059027777778, 0.2016059027777778, 0.201171875
# ]

# Samples for AER vs RESIDUAL (res: 32x32, frames: 10, range: (0,100), precision: 0.0001)
original_sample = [
    0.0185546875, 0.0185546875, 0.019097222222222224, 0.01953125,
    0.019314236111111112, 0.019205729166666668, 0.019205729166666668,
    0.019097222222222224, 0.018771701388888888, 0.019097222222222224,
    0.0185546875, 0.0185546875, 0.019097222222222224, 0.019097222222222224,
    0.018771701388888888, 0.019097222222222224, 0.019097222222222224,
    0.019097222222222224, 0.019097222222222224, 0.018880208333333332,
    0.019097222222222224, 0.019097222222222224, 0.019097222222222224,
    0.018771701388888888, 0.01953125, 0.01953125, 0.019422743055555556,
    0.019314236111111112, 0.019205729166666668, 0.019422743055555556,
    0.019314236111111112, 0.019748263888888888, 0.019639756944444444,
    0.01953125, 0.019314236111111112, 0.019422743055555556, 0.01953125,
    0.019097222222222224, 0.018880208333333332, 0.019097222222222224,
    0.019314236111111112, 0.019205729166666668, 0.018880208333333332,
    0.019205729166666668, 0.019314236111111112, 0.018880208333333332,
    0.019639756944444444, 0.019097222222222224, 0.019097222222222224,
    0.019314236111111112, 0.019422743055555556, 0.019639756944444444,
    0.019314236111111112, 0.019097222222222224, 0.019097222222222224,
    0.019314236111111112, 0.018446180555555556, 0.0185546875,
    0.019097222222222224, 0.019097222222222224, 0.018771701388888888,
    0.019205729166666668, 0.019314236111111112, 0.019097222222222224,
    0.018771701388888888, 0.019422743055555556, 0.018880208333333332,
    0.019422743055555556, 0.019097222222222224, 0.018771701388888888,
    0.018880208333333332, 0.019314236111111112, 0.019097222222222224,
    0.019314236111111112, 0.019097222222222224, 0.019314236111111112,
    0.019205729166666668, 0.019422743055555556, 0.019314236111111112,
    0.019205729166666668, 0.019097222222222224, 0.019205729166666668,
    0.01953125, 0.01953125, 0.019314236111111112, 0.019314236111111112,
    0.019422743055555556, 0.019097222222222224, 0.019097222222222224,
    0.019205729166666668, 0.019205729166666668, 0.018880208333333332,
    0.019205729166666668, 0.019422743055555556, 0.01953125,
    0.018771701388888888, 0.019097222222222224, 0.0185546875,
    0.019205729166666668, 0.019314236111111112
]

# sample_means = []
# for _ in range(10000):  #so B=10000
# 	sample = np.random.choice(original_sample, size=len(original_sample))
# 	sample_means.append(np.mean(sample))
B = 100000
print(f"Generating {B} samples")
bsamples = np.random.choice(original_sample, (len(original_sample), B))
bmeans = np.mean(bsamples, axis=0)

print(f"Sorting sample means...")
bmeans.sort()

print(f"Plotting histogram of sample means...")
plt.hist(bmeans)
plt.title("Histogram of sample means")
plt.show()

print(f"Estimated statistic value: {np.mean(bmeans)}")

conf_int = np.percentile(bmeans, [2.5, 97.5])
print(f"Confidence interval: {conf_int}")
