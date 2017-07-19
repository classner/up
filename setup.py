#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The setup script for the entire project.
@author: Christoph Lassner
"""
from setuptools import setup
from pip.req import parse_requirements
from setuptools.extension import Extension
import os.path as path
import numpy as np
import config

VERSION = '1.0'
REQS = [str(ir.req) for ir in parse_requirements('requirements.txt',
                                                 session='tmp')]

setup(
    name='up_tools',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    packages=['up_tools'],
    test_suite='tests',
    dependency_links=['http://github.com/classner/clustertools/tarball/master#egg=clustertools'],
    include_package_data=True,
    install_requires=REQS,
    version=VERSION,
    license='Creative Commons Non-Commercial 4.0',
)
