from setuptools import setup, find_packages

setup(
    name='EventCompression',
    version='1.0.0',
    url='https://github.com/ymykoliv/master_thesis',
    author='Yaroslav Mykoliv',
    author_email='yaroslav.mykoliv28@gmail.com',
    description='Master thesis',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
