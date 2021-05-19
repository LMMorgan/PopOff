from setuptools import setup
from popoff import __version__ as VERSION
import re

def remove_img_tags(data):
    p = re.compile(r'<img.*?/>')
    return p.sub('', data)

with open("README.md", "r") as fh:
    long_description = fh.read()

config = {"description": "POPOFF: POtential Parameter Optimisation for Force-Fields",
          "long_description": long_description,
          "long_description_content_type": "text/markdown",
          "name": "PopOff",
          "author": "Lucy M. Morgan",
          "author_email": "l.m.morgan@bath.ac.uk",
          "packages": ["popoff"],    #setuptools.find_packages()
          "url": "https://github.com/LMMorgan/PopOff",
          "version": VERSION,
          "install_requires": open( "requirements.txt" ).read(),
          "license": "MIT"}

setup(**config)
