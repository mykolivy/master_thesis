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


def log(msg, out):
    print(msg)
    out.write(f'{msg}\n')


# Define script interface
parser = argparse.ArgumentParser(description='Perform binary search of event \
        rate performance threshold')
parser.add_argument('sequence',
                    help='type of sequences generated',
                    choices=[
                        'moving_edge', 'random_pixel', 'single_color',
                        'checkers', 'rate_random_flip'
                    ])
parser.add_argument('-r',
                    '--res',
                    dest='res',
                    action='store',
                    default=[64, 64],
                    nargs=2,
                    type=int,
                    help='Resolution of the generated sequences: y x')
parser.add_argument('-f',
                    '--fps',
                    dest='fps',
                    action='store',
                    default=30,
                    help='Framerate of the generated sequences',
                    type=int)
parser.add_argument('-d',
                    '--duration',
                    dest='duration',
                    action='store',
                    type=int,
                    default=5,
                    help='Duration of the generated sequences')
parser.add_argument('--rate', type=float, default=0.5)
parser.add_argument('--value', type=int, default=0)
parser.add_argument('--range', type=int, default=[0, 255], nargs=2)
parser.add_argument('--precision', type=int, default=1)
parser.add_argument('--format', default='aer', choices=['aer', 'caer'])
parser.add_argument('--coder', default='lpaq1')
parser.add_argument('out', help='output file')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# Define filenames and parameter strings
temp_name = args.out.split('.')[:-1]
temp_name += ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=6))
temp_name = ''.join(temp_name)
args_res = f'{args.res[0]} {args.res[1]}'
args_range = f'{args.range[0]} {args.range[1]}'
precision_str = f'--precision {args.precision}'
params_str = f'-r {args_res} -f {args.fps} -d {args.duration} \
               --value {args.value} --range {args_range}'

rate_str = f'--rate {args.rate}'
raw_name = f'{temp_name}.raw'
frm_name = f'{temp_name}.{args.format}'

args.coder = ' '.join(args.coder.split('~'))
out_redir = '' if args.verbose else '> /dev/null 2>&1'

# Binary search of event performance threshold
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w+') as out:
    log(f"Parameters used: {args}\n", out)
    interval = [0, 1]
    raw_paq_size = args.precision + 1
    frm_paq_size = 0
    while abs(raw_paq_size - frm_paq_size) > args.precision:
        rate = interval[0] + (interval[1] - interval[0]) / 2
        rate_str = f'--rate {rate}'
        log(f'rate: {rate}', out)

        os.system(f'./synthetic.py {params_str} {rate_str} {args.sequence}\
                {raw_name} {out_redir}')
        size = os.path.getsize(f'{raw_name}')
        log(f'    raw size: {size}', out)

        os.system(f'./synthetic.py {params_str} {rate_str} {args.sequence}\
                {frm_name} {out_redir}')
        size = os.path.getsize(f'{frm_name}')
        log(f'    {args.format} size: {size}', out)

        os.system(f'{args.coder} {raw_name} {raw_name}.paq {out_redir}')
        raw_paq_size = os.path.getsize(f'{raw_name}.paq')
        log(f'    raw.paq size: {raw_paq_size}', out)

        os.system(f'{args.coder} {frm_name} {frm_name}.paq {out_redir}')
        frm_paq_size = os.path.getsize(f'{frm_name}.paq')
        log(f'    {args.format}.paq size: {frm_paq_size}', out)

        diff = abs(raw_paq_size - frm_paq_size)
        log(f'    diff: {diff}', out)
        log('', out)

        if raw_paq_size > frm_paq_size:
            interval[0] = interval[0] + (interval[1] - interval[0]) / 2
        else:
            interval[1] = interval[0] + (interval[1] - interval[0]) / 2

        os.system(f'rm *{temp_name}*')

        prev_rate = rate
