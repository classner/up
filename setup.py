#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The setup script for the entire project.
@author: Christoph Lassner
"""
from setuptools import setup
from pip.req import parse_requirements
from setuptools.extension import Extension
from Cython.Build import cythonize
import os.path as path
import numpy as np
import config

VERSION = '0.1'
REQS = [str(ir.req) for ir in parse_requirements('requirements.txt',
                                                 session='tmp')]
EXT_MODULES = [Extension("up_tools.sample2meshdist",
                         ["up_tools/sample2meshdist.pyx"],
                         include_dirs=[np.get_include(),
                                       config.EIGEN_FP],
                         language="c++"),
               Extension("up_tools.fast_derivatives.smpl_derivatives",
                         ["up_tools/fast_derivatives/smpl_derivatives.pyx",
                          "up_tools/fast_derivatives/src/derivatives_wrt_pose.cpp",
                          "up_tools/fast_derivatives/src/derivatives_wrt_shape.cpp",
                          "up_tools/fast_derivatives/src/homogeneous_transform_matrix.cpp",
                          "up_tools/fast_derivatives/src/kinematic_tree.cpp",
                          "up_tools/fast_derivatives/src/rodrigues.cpp"],
                         include_dirs=[np.get_include(),
                                       path.abspath(path.join(path.dirname(__file__),
                                                              'up_tools',
                                                              'fast_derivatives',
                                                              'include'))],
                         language="c++")
               ]

setup(
    name='up_tools',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    packages=['up_tools'],
    ext_modules=cythonize(EXT_MODULES),
    test_suite='tests',
    dependency_links=['http://github.com/classner/clustertools/tarball/master#egg=clustertools'],
    include_package_data=True,
    install_requires=REQS,
    version=VERSION,
    license='Creative Commons Non-Commercial 4.0',
)
