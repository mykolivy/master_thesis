#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys
import os
import subprocess
import shutil
import random
import events
import argparse
import string


def log(msg, out, end='\n'):
	print(msg, end=end, flush=True)
	out.write(f'{msg}{end}')


# Binary search of event performance threshold
def threshold_binary_search(params_str, sequence, coder, ext, raw_name,
                            frm_name, iterations, out_redir):
	avg_result = 0
	for it in range(0, iterations):
		interval = [0, 1]
		raw_paq_size = args.precision + 1
		frm_paq_size = 0

		rate = interval[0] + (interval[1] - interval[0]) / 2
		prev_rate = rate + args.precision + 1
		while abs(rate - prev_rate) > args.precision and \
          abs(raw_paq_size - frm_paq_size) != 0:
			rate_str = f'--rate {rate}'
			#log(f'          {rate}', out)

			os.system(f'./synthetic.py {params_str} {rate_str} {sequence}\
                    {raw_name} {out_redir}')
			size = os.path.getsize(f'{raw_name}')

			os.system(f'./synthetic.py {params_str} {rate_str} {sequence}\
                    {frm_name} {out_redir}')
			size = os.path.getsize(f'{frm_name}')

			os.system(f'{args.coder} {raw_name} {raw_name}.paq {out_redir}')
			raw_paq_size = os.path.getsize(f'{raw_name}.paq')

			os.system(f'{args.coder} {frm_name} {frm_name}.paq {out_redir}')
			frm_paq_size = os.path.getsize(f'{frm_name}.paq')

			diff = abs(raw_paq_size - frm_paq_size)

			if raw_paq_size > frm_paq_size:
				interval[0] = interval[0] + (interval[1] - interval[0]) / 2
			else:
				interval[1] = interval[0] + (interval[1] - interval[0]) / 2

			os.system(f'rm -rf *{temp_name}*')

			prev_rate = rate
			rate = interval[0] + (interval[1] - interval[0]) / 2
		avg_result += rate
	return avg_result * (1.0 / iterations)


def main():
	# Define script interface
	parser = argparse.ArgumentParser(description='Perform binary search of event \
					rate performance threshold')
	parser.add_argument('sequence',
	                    help='type of sequences generated',
	                    choices=[
	                        'moving_edge', 'random_pixel', 'single_color',
	                        'checkers', 'rate_random_flip', 'rate_random_change',
	                        'rate_random'
	                    ])
	parser.add_argument('--precision', type=int, default=1)
	parser.add_argument('--format',
	                    default='aer',
	                    choices=['aer', 'caer', 'aer_true'])
	parser.add_argument('--coder', default='lpaq1')
	parser.add_argument('-d',
	                    dest='durations',
	                    type=int,
	                    nargs='+',
	                    default=[1, 2, 4, 8, 16])
	parser.add_argument('-r',
	                    dest='resolutions',
	                    type=int,
	                    nargs='+',
	                    default=[1, 2, 4, 8, 16, 32, 64])
	parser.add_argument('-i', dest='iterations', type=int, default=1)
	parser.add_argument('out', help='output file')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('-b', dest='batch_size', type=int, default=1)
	args = parser.parse_args()

	args.coder = ' '.join(args.coder.split('~'))
	out_redir = '' if args.verbose else '> /dev/null 2>&1'

	args.precision = 0.00001

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w+') as out:
		log(f"Parameters used: {args}\n", out)
		log(f"RES, DURATION: THRESHOLD", out)
		results = np.zeros((len(args.resolutions), len(args.durations)))
		for i, res in enumerate(args.resolutions):
			for j, dur in enumerate(args.durations):
				log(f"{res}, {dur}:      ", out, end='')
				for sample in range(args.batch_size):
					done = sample * 100 / args.batch_size
					print(f"\b\b\b\b\b{done:4.1f}%", end='', flush=True)
					temp_name = args.out.split('.')[:-1]
					temp_name += ''.join(
					    random.choices(string.ascii_uppercase + string.digits, k=6))
					temp_name = ''.join(temp_name)
					args_res = f'{res} {res}'
					precision_str = f'--precision {args.precision}'
					params_str = f'-r {args_res} -d {dur}'
					raw_name = f'{temp_name}.raw'
					frm_name = f'{temp_name}.{args.format}'

					rate = threshold_binary_search(params_str, args.sequence, args.coder,
					                               args.format, raw_name, frm_name,
					                               args.iterations, out_redir)
					results[i][j] += rate

				results[i][j] /= args.batch_size
				log(f"\b\b\b\b\b{results[i][j]:8.6f}", out)
		log('', out)
		log(f"TABLE:\n{results}\n", out)
