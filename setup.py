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

setup(
    name=dist_name,
    author=author,
    author_email=email,
    url=url,
    version="1.0.0",
    packages=find_packages(),
    package_dir={dist_name: package_name},
    include_package_data=True,
    license='MIT',
    description=description,
    long_description=Path('README.md').read_text() if Path('README.md').exists() else '',
    long_description_content_type="text/markdown",
    install_requires=[
        'jax==0.4.33',
        'numpy',
        'optax==0.2.3',
        'scipy==1.14.1',
        'matplotlib'
    ],
    python_requires=">=3.10",
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    platforms=['ALL'],
    py_modules=[package_name],
)