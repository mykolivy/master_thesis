# import tensorflow as tf
import tensorflow_compression as tfc
import pdb
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
# SEQ_LEN = 128
# x = tf.random.normal((100, SEQ_LEN, 2))

# analysis = tf.keras.Sequential([
#     tfc.SignalConv1D(32, 3, strides_down=2, padding="same_zeros"),
#     tfc.SignalConv1D(32, 3, strides_down=2, padding="same_zeros"),
#     tfc.SignalConv1D(32, 3, strides_down=2, padding="same_zeros")
# ])
# synthesis = tf.keras.Sequential([
#     tfc.SignalConv1D(32, 3, strides_up=2, padding="same_zeros"),
#     tfc.SignalConv1D(32, 3, strides_up=2, padding="same_zeros"),
#     tfc.SignalConv1D(32, 3, strides_up=2, padding="same_zeros"),
#     tfc.SignalConv1D(2, 3, padding="same_zeros"),
# ])
# analised = analysis(x)
# pdb.set_trace()
# reconstr = synthesis(analised)
# pdb.set_trace()

# record_defaults = [float()] * 6
filenames = ["/home/dumpling/Documents/uni/thesis/test.txt"]
# # dataset = tf.data.experimental.CsvDataset(filenames,
# #                                           record_defaults=record_defaults,
# #                                           field_delim=' ').batch(4)
# # for x in dataset:
# # 	print(tf.reshape(tf.transpose(tf.stack(x)), (4, 2, -1)))
# # 	break


def string_to_tensor(x):
	split = tf.strings.split(x, sep=' ').values
	numbers = tf.strings.to_number(split)
	return tf.reshape(numbers, (-1, 2, 1))


print("ANOTHER")
dataset = tf.data.TextLineDataset(
    filenames, compression_type=None, buffer_size=None,
    num_parallel_reads=None).map(string_to_tensor).shuffle(10000).batch(2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
print(next_element)
with tf.Session() as sess:
	for i in range(3):
		print(sess.run(next_element))