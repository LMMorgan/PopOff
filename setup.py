from setuptools import setup
from lammps_potenial_fitting import __version__ as VERSION
import re

def remove_img_tags(data):
    p = re.compile(r'<img.*?/>')
    return p.sub('', data)

with open("README.md", "r") as fh:
    long_description = fh.read()

config = {"description": "Modular potential fitting code for classical MD buckingham potentials",
          "long_description": long_description,
          "long_description_content_type": "text/markdown",
          "name": "lammps_potenial_fitting",
          "author": "Lucy M. Morgan",
          "author_email": "l.m.morgan@bath.ac.uk",
          "packages": ["lammps_potenial_fitting"],    #setuptools.find_packages()
          "url": "https://github.com/LMMorgan/lammps_potenial_fitting",
          "version": VERSION,
          "install_requires": open( "requirements.txt" ).read(),
          "license": "MIT"}

setup(**config)
