from setuptools import setup
from buckfit import __version__ as VERSION
import re

def remove_img_tags(data):
    p = re.compile(r'<img.*?/>')
    return p.sub('', data)

with open("README.md", "r") as fh:
    long_description = fh.read()

config = {"description": "Modular potential fitting code for classical MD buckingham potentials",
          "long_description": long_description,
          "long_description_content_type": "text/markdown",
          "name": "BuckFit",
          "author": "Lucy M. Morgan",
          "author_email": "l.m.morgan@bath.ac.uk",
          "packages": ["buckfit"],    #setuptools.find_packages()
          "url": "https://github.com/LMMorgan/BuckFit",
          "version": VERSION,
          "install_requires": open( "requirements.txt" ).read(),
          "license": "MIT"}

setup(**config)
