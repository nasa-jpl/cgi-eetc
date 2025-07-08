# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# -*- coding: utf-8 -*-
import logging
import sys
from io import open
from os import path

import eetc

try:
    from setuptools import setup, find_packages
except ImportError:
    logging.exception('Please install or upgrade setuptools or pip')
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

setup(
    name='eetc',
    version=eetc.__version__,
    description='A package for estimating exposure times for CGI calibration targets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/eetc',
    author='Sam Halverson, Sam Miller, Kevin Ludwick, Eric Cady',
    author_email='samuel.halverson@jpl.nasa.gov, sam.miller@uah.edu, kjl0025@uah.edu, eric.j.cady@jpl.nasa.gov',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    package_data={
        'eetc': [
            '*',
            '*.yaml',
            'ut_config/*',
            'config/*',
            'thptcurves/*',
            'thptcurves/dpam/*',
            'thptcurves/fpam/*',
            'thptcurves/fsam/*',
            'thptcurves/qe/*',
            'thptcurves/spam_lsam/*',
            'thptcurves/static_optics/*',
            'ut_thptcurves/*',
            'ut_thptcurves/dpam/*',
            'ut_thptcurves/fpam/*',
            'ut_thptcurves/fsam/*',
            'ut_thptcurves/qe/*',
            'ut_thptcurves/spam_lsam/*',
            'ut_thptcurves/static_optics/*',
            'flux_grid_generation/*',
            'flux_grid_generation/astro_filters/*',
            'flux_grid_generation/bpgs_atlas_csv/*',
            'flux_grid_generation/cfam_filter_curves/*',
            'flux_grid_generation/config/*',
            'flux_grid_generation/grid_files/*',
            'flux_grid_generation/ut_bpgs_atlas_csv/UT_SPT.txt',
            'flux_grid_generation/ut_cfam_filter_curves/UT.csv',
            'flux_grid_generation/ut_bpgs_atlas_csv_err/*',
            'flux_grid_generation/ut_cfam_filter_curves_err/*',
            'flux_grid_generation/ut_config/*',
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements
)
