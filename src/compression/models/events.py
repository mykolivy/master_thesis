# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

import argparse
import glob
import sys
import pdb
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2

import tensorflow_compression as tfc

tf.disable_eager_execution()


def string_to_tensor(x):
	split = tf.strings.split(x, sep=' ').values
	numbers = tf.strings.to_number(split)
	return tf.reshape(numbers, (-1, 2))


def read_png(filename):
	"""Loads a PNG image file."""
	string = tf.read_file(filename)
	image = tf.image.decode_image(string, channels=1)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, (28, 28, 1))
	image /= 255
	return tf.random.normal((32, 32, 3))


def read_events(filename):
	df = pd.read_csv(filename, sep=' ', dtype=np.float32)
	values = np.transpose(np.reshape(df.values, (-1, 2, 128)), (0, 2, 1))
	return values


def quantize_image(image):
	image = tf.round(image * 255)
	image = tf.saturate_cast(image, tf.uint8)
	return image


def write_png(filename, image):
	"""Saves an image to a PNG file."""
	image = quantize_image(image)
	string = tf.image.encode_png(image)
	return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
	"""The analysis transform."""
	def __init__(self, num_filters, *args, **kwargs):
		self.num_filters = num_filters
		super(AnalysisTransform, self).__init__(*args, **kwargs)

	def build(self, input_shape):
		self._layers = [
		    tfc.SignalConv1D(self.num_filters,
		                     3,
		                     strides_down=2,
		                     padding="same_zeros"),
		    tfc.GDN(),
		    tfc.SignalConv1D(self.num_filters, 3, padding="same_zeros"),
		    tf.keras.layers.ReLU(),
		    tfc.SignalConv1D(self.num_filters,
		                     3,
		                     strides_down=2,
		                     padding="same_zeros"),
		    tfc.GDN(),
		    tfc.SignalConv1D(self.num_filters, 3, padding="same_zeros"),
		    tf.keras.layers.ReLU(),
		    tfc.SignalConv1D(self.num_filters // 8,
		                     3,
		                     strides_down=2,
		                     padding="same_zeros"),
		    tfc.GDN()
		]
		super(AnalysisTransform, self).build(input_shape)

	def call(self, tensor):
		for layer in self._layers:
			tensor = layer(tensor)
		return tensor


class SynthesisTransform(tf.keras.layers.Layer):
	"""The synthesis transform."""
	def __init__(self, num_filters, *args, **kwargs):
		self.num_filters = num_filters
		super(SynthesisTransform, self).__init__(*args, **kwargs)

	def build(self, input_shape):
		self._layers = [
		    tfc.GDN(inverse=True),
		    tfc.SignalConv1D(self.num_filters // 8,
		                     3,
		                     strides_up=2,
		                     padding="same_zeros"),
		    tf.keras.layers.ReLU(),
		    tfc.SignalConv1D(self.num_filters, 3, padding="same_zeros"),
		    tfc.GDN(inverse=True),
		    tfc.SignalConv1D(self.num_filters,
		                     3,
		                     strides_up=2,
		                     padding="same_zeros"),
		    tf.keras.layers.ReLU(),
		    tfc.SignalConv1D(self.num_filters, 3, padding="same_zeros"),
		    tfc.GDN(inverse=True),
		    tfc.SignalConv1D(self.num_filters,
		                     3,
		                     strides_up=2,
		                     padding="same_zeros"),
		    tfc.SignalConv1D(2, 3, padding="same_zeros"),
		]

		super(SynthesisTransform, self).build(input_shape)

	def call(self, tensor):
		for layer in self._layers:
			tensor = layer(tensor)
		return tensor


def train(args):
	"""Trains the model."""

	if args.verbose:
		tf.logging.set_verbosity(tf.logging.INFO)

	# Create input data pipeline.
	with tf.device("/cpu:0"):
		train_files = glob.glob(args.train_glob)[:3]
		if not train_files:
			raise RuntimeError("No training images found with glob '{}'.".format(
			    args.train_glob))
		train_dataset = tf.data.TextLineDataset(
		    train_files,
		    compression_type=None,
		    buffer_size=len(train_files),
		    num_parallel_reads=args.preprocess_threads)
		train_dataset = train_dataset.map(
		    string_to_tensor, num_parallel_calls=args.preprocess_threads)
		train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
		train_dataset = train_dataset.batch(args.batchsize)
		train_dataset = train_dataset.prefetch(32)

	num_pixels = args.batchsize * 128

	# Get training patch from dataset.
	x = train_dataset.make_one_shot_iterator().get_next()

	# Instantiate model.
	analysis_transform = AnalysisTransform(32)
	entropy_bottleneck = tfc.EntropyBottleneck()
	synthesis_transform = SynthesisTransform(32)

	# Build autoencoder.
	y = analysis_transform(x)
	y_tilde, likelihoods = entropy_bottleneck(y, training=True)
	x_tilde = synthesis_transform(y_tilde)
	timestamps, polarities = tf.split(x_tilde, num_or_size_splits=2, axis=-1)
	timestamps = tf.math.abs(timestamps)
	polarities = tf.math.tanh(polarities)
	x_tilde = tf.concat([timestamps, polarities], axis=-1)

	train_bpp = tf.reduce_mean(
	    -tf.reduce_sum(likelihoods * tf.log(likelihoods), axis=[1, 2]) /
	    np.log(2))

	# Mean squared error across pixels.
	train_mse = tf.reduce_mean((x - x_tilde)**2.)

	# The rate-distortion cost.
	train_loss = args.lmbda * train_mse + train_bpp

	# Minimize loss and auxiliary loss, and execute update op.
	step = tf.train.create_global_step()
	main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
	main_step = main_optimizer.minimize(train_loss, global_step=step)

	aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
	aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

	train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

	tf.summary.scalar("loss", train_loss)
	tf.summary.scalar("bpp", train_bpp)
	tf.summary.scalar("mse", train_mse)

	hooks = [
	    tf.train.StopAtStepHook(last_step=args.last_step),
	    tf.train.NanTensorHook(train_loss),
	]
	with tf.train.MonitoredTrainingSession(hooks=hooks,
	                                       checkpoint_dir=args.checkpoint_dir,
	                                       save_checkpoint_secs=300,
	                                       save_summaries_secs=60) as sess:
		while not sess.should_stop():
			sess.run(train_op)


def compress(args):
	"""Compresses an event file."""

	x = tf.constant(read_events(args.input_file))
	x_shape = tf.shape(x)

	analysis_transform = AnalysisTransform(32)
	entropy_bottleneck = tfc.EntropyBottleneck()
	synthesis_transform = SynthesisTransform(32)

	y = analysis_transform(x)
	string = entropy_bottleneck.compress(y)

	y_hat, likelihoods = entropy_bottleneck(y, training=False)
	x_hat = synthesis_transform(y_hat)

	timestamps, polarities = tf.split(x_hat, num_or_size_splits=2, axis=-1)
	timestamps = tf.math.abs(timestamps)
	polarities = tf.round(tf.math.tanh(polarities))
	x_hat = tf.concat([timestamps, polarities], axis=-1)

	eval_bpp = tf.reduce_mean(
	    -tf.reduce_sum(likelihoods * tf.log(likelihoods), axis=[1, 2]) /
	    np.log(2))

	mse = tf.reduce_mean((x - x_hat)**2.)

	with tf.Session() as sess:
		# Load the latest model checkpoint, get the compressed string and the tensor
		# shapes.
		latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
		tf.train.Saver().restore(sess, save_path=latest)
		tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
		arrays = sess.run(tensors)

		# Write a binary file with the shape information and the compressed string.
		packed = tfc.PackedTensors()
		packed.pack(tensors, arrays)
		with open(args.output_file, "wb") as f:
			f.write(packed.string)

		# If requested, transform the quantized image back and measure performance.
		if args.verbose:
			# eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
			# [eval_bpp, mse, psnr, msssim, num_pixels])
			eval_bpp, mse = sess.run([eval_bpp, mse])

			compression_ratio = os.path.getsize(args.input_file) / len(packed.string)

			print("Mean squared error: {:0.4f}".format(mse))
			print("Estimated entropy: {}".format(eval_bpp))
			print("Compression ratio: {}".format(compression_ratio))


def decompress(args):
	"""Decompresses an image."""

	# Read the shape information and compressed string from the binary file.
	string = tf.placeholder(tf.string, [1])
	x_shape = tf.placeholder(tf.int32, [2])
	y_shape = tf.placeholder(tf.int32, [2])
	with open(args.input_file, "rb") as f:
		packed = tfc.PackedTensors(f.read())
	tensors = [string, x_shape, y_shape]
	arrays = packed.unpack(tensors)

	# Instantiate model.
	entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
	synthesis_transform = SynthesisTransform(args.num_filters)

	# Decompress and transform the image back.
	y_shape = tf.concat([y_shape, [args.num_filters]], axis=0)
	y_hat = entropy_bottleneck.decompress(string,
	                                      y_shape,
	                                      channels=args.num_filters)
	x_hat = synthesis_transform(y_hat)

	# Remove batch dimension, and crop away any extraneous padding on the bottom
	# or right boundaries.
	x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

	# Write reconstructed image out as a PNG file.
	op = write_png(args.output_file, x_hat)

	# Load the latest model checkpoint, and perform the above actions.
	with tf.Session() as sess:
		latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
		tf.train.Saver().restore(sess, save_path=latest)
		sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
	"""Parses command line arguments."""
	parser = argparse_flags.ArgumentParser(
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# High-level options.
	parser.add_argument(
	    "--verbose",
	    "-V",
	    action="store_true",
	    help="Report bitrate and distortion when training or compressing.")
	parser.add_argument("--num_filters",
	                    type=int,
	                    default=128,
	                    help="Number of filters per layer.")
	parser.add_argument("--checkpoint_dir",
	                    default="train",
	                    help="Directory where to save/load model checkpoints.")
	subparsers = parser.add_subparsers(
	    title="commands",
	    dest="command",
	    help="What to do: 'train' loads training data and trains (or continues "
	    "to train) a new model. 'compress' reads an image file (lossless "
	    "PNG format) and writes a compressed binary file. 'decompress' "
	    "reads a binary file and reconstructs the image (in PNG format). "
	    "input and output filenames need to be provided for the latter "
	    "two options. Invoke '<command> -h' for more information.")

	# 'train' subcommand.
	train_cmd = subparsers.add_parser(
	    "train",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Trains (or continues to train) a new model.")
	train_cmd.add_argument(
	    "--train_glob",
	    default="images/*.png",
	    help="Glob pattern identifying training data. This pattern must expand "
	    "to a list of RGB images in PNG format.")
	train_cmd.add_argument("--batchsize",
	                       type=int,
	                       default=8,
	                       help="Batch size for training.")
	train_cmd.add_argument("--patchsize",
	                       type=int,
	                       default=256,
	                       help="Size of image patches for training.")
	train_cmd.add_argument("--lambda",
	                       type=float,
	                       default=0.01,
	                       dest="lmbda",
	                       help="Lambda for rate-distortion tradeoff.")
	train_cmd.add_argument("--last_step",
	                       type=int,
	                       default=1000000,
	                       help="Train up to this number of steps.")
	train_cmd.add_argument(
	    "--preprocess_threads",
	    type=int,
	    default=16,
	    help="Number of CPU threads to use for parallel decoding of training "
	    "images.")

	# 'compress' subcommand.
	compress_cmd = subparsers.add_parser(
	    "compress",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Reads a PNG file, compresses it, and writes a TFCI file.")

	# 'decompress' subcommand.
	decompress_cmd = subparsers.add_parser(
	    "decompress",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Reads a TFCI file, reconstructs the image, and writes back "
	    "a PNG file.")

	# Arguments for both 'compress' and 'decompress'.
	for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
		cmd.add_argument("input_file", help="Input filename.")
		cmd.add_argument(
		    "output_file",
		    nargs="?",
		    help="Output filename (optional). If not provided, appends '{}' to "
		    "the input filename.".format(ext))

	# Parse arguments.
	args = parser.parse_args(argv[1:])
	if args.command is None:
		parser.print_usage()
		sys.exit(2)
	return args


def main(args):
	# Invoke subcommand.
	if args.command == "train":
		train(args)
	elif args.command == "compress":
		if not args.output_file:
			args.output_file = args.input_file + ".tfci"
		compress(args)
	elif args.command == "decompress":
		if not args.output_file:
			args.output_file = args.input_file + ".png"
		decompress(args)


if __name__ == "__main__":
	app.run(main, flags_parser=parse_args)
