import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from event_compression.codec import codecs
from event_compression.sequence import sequences

codecs = list(codecs().keys()).append("raw")
sequences = list(sequences().keys())


def get_parser(script_path):
	"""
	Instantiate a parser with arguments defined in standard location.

	**Arguments**:
		
		``script_path``: path to script for which to instantiate parser.

	**Return**:

		Standard parser for script under ``script_path``.

	**Example**:

		In your script file (in src/scripts)::

			import util

			parser = util.get_parser(__file__)

	"""
	path = Path(script_path)
	path = path.parent / "args" / (path.stem + '.json')
	return parser_from_file(path)


def parser_from_file(path):
	"""
	Instantiate a parser with arguments defined in file under ``path``.

	**Arguments**:
		
		``path``: path to a .json file with argument definitions.

	**Return**:

		Parser.

	**Example**::

		from pathlib import Path

		# Get relative path to the script being executed
		path = Path(__file__).parent / "args/script.json"

		parser = parser_from_file(path)
	"""
	with open(path) as f:
		data = json.load(f)
		parser = argparse.ArgumentParser(description=data["description"])
		_add_arguments(parser, data["args"])

	return parser


def _add_arguments(parser, args):
	for arg in args:
		for item in arg["kwargs"].items():
			if isinstance(item[1], str):
				value = item[1].strip()
				if value[0] == '@':
					arg["kwargs"][item[0]] = eval(value[1:])

		parser.add_argument(*arg["args"], **arg["kwargs"])


def log_result():
	def decorate(f):
		def g(*args, **vargs):
			result = f(*args, **vargs)
			print(result)
			return result

		return g

	return decorate


def log(msg, out, end='\n'):
	print(msg, end=end, flush=True)
	out.write(f'{msg}{end}')
