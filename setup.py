import os
from glob import glob
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

exec(open('graphnmr/version.py').read())

setup(name = 'graphnmr', 
      version = __version__,
      description = 'GCN NMR Predictor',
      long_description=readme(),
      author = 'Andrew White', 
      author_email = 'andrew.white@rochester.edu', 
      url = 'http://thewhitelab.org/Software',
      license = 'TBA',
      packages = ['graphnmr'],
      install_requires=['numpy', 'matplotlib', 'gsd', 'networkx', 'pygraphviz', 'tqdm'])
