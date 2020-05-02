import pytest, operator, functools, tempfile, os
import numpy as np
from event_compression.codec.aer import AER
from event_compression.sequence.synthetic import RandomChange, Config


def seq_to_bytes(seq):
	result = bytearray()

	for frame in iter(seq):
		assert frame.dtype == 'uint8'
		result += frame.tobytes()

	return result


def entropy_compress(coder, data):
	with tempfile.NamedTemporaryFile('w+b') as raw:
		with tempfile.NamedTemporaryFile('w+b') as baseline:
			raw.write(data)
			raw.flush()

			os.system(f"{coder} {raw.name} {baseline.name}")

			with open(baseline.name, "rb") as f:
				return f.read()
			#return os.path.getsize(baseline.name)


def entropy_decompress(decoder, data):
	with tempfile.NamedTemporaryFile('w+b') as encoded:
		with tempfile.NamedTemporaryFile('w+b') as decoded:
			encoded.write(data)
			encoded.flush()

			os.system(f"{decoder} {encoded.name} {decoded.name}")

			with open(decoded.name, 'rb') as f:
				return f.read()


def check_result(seq, codec, negative=False):
	coder = "src/event_compression/scripts/bin/lpaq1 0"
	decoder = "src/event_compression/scripts/bin/lpaq1 d"

	frames1 = [x.copy() for x in iter(seq)]
	frames2 = [x.copy() for x in iter(seq)]

	if not all([np.equal(x, y).all() for x, y in zip(frames1, frames2)]):
		return False

	raw_data = seq_to_bytes(frames1)
	encoded = functools.reduce(operator.add, codec.encoder(frames2), bytearray())

	bpaq = entropy_compress(coder, raw_data)
	paq = entropy_compress(coder, encoded)

	bsize = len(bpaq)
	size = len(paq)

	# this is the main test of compressed sizes
	condition = size < bsize
	print(negative)
	if (not (negative or condition)) or (negative and condition):
		return False

	# Test correct decompression
	raw_decoded = entropy_decompress(decoder, bpaq)
	aer_decoded = entropy_decompress(decoder, paq)
	decoded = [x.copy() for x in codec.decoder(aer_decoded)]

	if raw_data != raw_decoded:
		return False

	if not all([np.equal(x, y).all() for x, y in zip(frames2, decoded)]):
		return False

	return True


class TestThresholdSearch:
	def test_aer_resolutions(self):
		"""
	                                    SUMMARY                                     
     Resolution            Frames                      Threshold                
================================================================================
       (1, 1)              10000                     0.05550555056              
       (2, 2)               2500                      0.3225090036              
       (4, 4)               625                       0.2666266026              
       (8, 8)               157                       0.2396233974              
      (16, 16)               40                       0.1547475962              
      (32, 32)               10                       0.1750759549              
      (64, 64)               3                        0.2606689453              
     (128, 128)              2                        0.3762268066              
     (256, 256)              2                        0.4510757446              
     (512, 512)              2                        0.4868911743              
    (1024, 1024)             2                        0.5026011467              
================================================================================
		"""
		resolutions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
		frames = [10000, 2500, 625, 157, 40, 10, 3, 2, 2, 2, 2]
		satis_rates = [
		    0.05, 0.3, 0.2, 0.23, 0.15, 0.17, 0.25, 0.37, 0.447, 0.48, 0.497
		]
		unsatis_rates = [x + 0.1 for x in satis_rates]

		assert len(resolutions) == len(satis_rates)
		assert len(resolutions) == len(unsatis_rates)

		results = {x: [False, False] for x in resolutions}

		for res, num_frames, srate, nrate in zip(resolutions, frames, satis_rates,
		                                         unsatis_rates):
			codec = AER

			config = Config((res, res), 1, num_frames, rate=srate)
			seq = RandomChange(config)
			results[res][0] = check_result(seq, codec)

			config = Config((res, res), 1, num_frames, rate=nrate)
			seq = RandomChange(config)
			results[res][1] = check_result(seq, codec, negative=True)

		print(results)
		assert all([all(x) for x in results.values()])
