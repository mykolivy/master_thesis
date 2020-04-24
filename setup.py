#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='event_compression',
      version='0.0.1',
      description='Event-based compression library',
      keywords=['compression', 'event', 'machine learning'],
      license='',
      author='Yaroslav Mykoliv',
      author_email='yaroslav.mykoliv28@gmail.com',
      long_description='',
      long_description_content_type='text/x-rst',
      url='',
      maintainer='Yaroslav Mykoliv',
      maintainer_email='yaroslav.mykoliv28@gmail.com',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data={
          "": ["*.json"],
          "scripts": ["args"]
      },
      include_package_data=True,
      entry_points={
          "console_scripts":
          ["synthetic = event_compression.scripts.synthetic:main"]
      },
      install_requires=[
          'pytest', 'numpy', 'tensorflow', 'sphinx>=3', 'sphinx-rtd-theme',
          'Pillow', 'opencv-python', 'tabulate'
      ])
