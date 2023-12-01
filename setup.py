#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from pathlib import Path
import re

from setuptools import setup, find_packages

author = 'artificial-scientist-lab'
email = 'carla.rodriguez@mpl.mpg.de, soeren.arlt@mpl.mpg.de, mario.krenn@mpl.mpg.de,'
description = 'XLuminA: An Auto-differentiating Discovery Framework for Super-Resolution Microscopy'
dist_name = 'xlumina'
package_name = 'xlumina'
year = '2023'
url = 'https://github.com/artificial-scientist-lab/XLuminA'


def get_version():
    content = open(Path(package_name) / '__init__.py').readlines()
    return "0.0.1"

setup(
    name=dist_name,
    author=author,
    author_email=email,
    url=url,
    version=get_version(),
    packages=find_packages(),
    package_dir={dist_name: package_name},
    include_package_data=True,
    license='MIT',
    description=description,
    long_description=open('README.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=[
        "jax==0.4.13",
        "jaxlib==0.4.13=+cuda11.cudnn86"
        "numpy>= 1.24.2",
        "optax==0.1.7", 
        "scipy==1.10.1", 
        "matplotlib"
    ],
    python_requires=">=3.8",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
    py_modules=[package_name],
)