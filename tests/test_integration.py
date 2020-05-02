import pytest, operator, functools, tempfile, os
import numpy as np
from event_compression.codec.aer import AER
from event_compression.codec.caer import CAER
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
		result_rates = [
		    0.05550555056, 0.3225090036, 0.2666266026, 0.2396233974, 0.1547475962,
		    0.1750759549, 0.2606689453, 0.3762268066, 0.4510757446, 0.4868911743,
		    0.5026011467
		]

		resolutions = resolutions[:-2]
		frames = frames[:-2]
		result_rates = result_rates[:-2]

		srates = [x - 0.01 for x in result_rates]
		nrates = [x + 0.05 for x in result_rates]

		codec = AER

		assert len(resolutions) == len(srates)
		assert len(resolutions) == len(nrates)

		results = {x: [False, False] for x in resolutions}

		for res, n_frames, srate, nrate in zip(resolutions, frames, srates, nrates):
			config = Config((res, res), 1, n_frames, rate=srate)
			seq = RandomChange(config)
			results[res][0] = check_result(seq, codec)

			config = Config((res, res), 1, n_frames, rate=nrate)
			seq = RandomChange(config)
			results[res][1] = check_result(seq, codec, negative=True)

		print(results)
		assert all([all(x) for x in results.values()])

	def test_caer_resolutions(self):
		"""
	                                    SUMMARY                                     
     Resolution            Frames                      Threshold                
================================================================================
       (1, 1)              10000                     0.06589658966              
       (2, 2)               2500                      0.3960184074              
       (4, 4)               625                       0.3931590545              
       (8, 8)               157                       0.4565504808              
      (16, 16)               40                       0.5116586538              
      (32, 32)               10                       0.588140191              
      (64, 64)               3                        0.6868408203              
     (128, 128)              2                        0.8062927246              
     (256, 256)              2                        0.828968811              
     (512, 512)              2                        0.8326633453              
    (1024, 1024)             2                        0.8334960938              
================================================================================
		"""
		resolutions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
		frames = [10000, 2500, 625, 157, 40, 10, 3, 2, 2, 2, 2]
		result_rates = [
		    0.06589658966, 0.3960184074, 0.3931590545, 0.4565504808, 0.5116586538,
		    0.588140191, 0.6868408203, 0.8062927246, 0.828968811, 0.8326633453,
		    0.8334960938
		]

		resolutions = resolutions[:-2]
		frames = frames[:-2]
		result_rates = result_rates[:-2]

		srates = [x - 0.01 for x in result_rates]
		nrates = [x + 0.01 for x in result_rates]

		codec = CAER

		assert len(resolutions) == len(srates)
		assert len(resolutions) == len(nrates)

		results = {x: [False, False] for x in resolutions}

		for res, n_frames, srate, nrate in zip(resolutions, frames, srates, nrates):
			config = Config((res, res), 1, n_frames, rate=srate)
			seq = RandomChange(config)
			results[res][0] = check_result(seq, codec)

			config = Config((res, res), 1, n_frames, rate=nrate)
			seq = RandomChange(config)
			results[res][1] = check_result(seq, codec, negative=True)

		print(results)
		assert all([all(x) for x in results.values()])
