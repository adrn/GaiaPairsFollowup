#!/usr/bin/env python

import os
import sys
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# package before the dependencies are installed.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__COMOVINGRV_SETUP__ = True
import comoving_rv

setup(
    name="comoving_rv",
    version=comoving_rv.__version__,
    author="Adrian Price-Whelan",
    author_email="adrianmpw@gmail.com",
    url="https://github.com/adrn/GaiaPairsFollowup",
    packages=["comoving_rv", "comoving_rv.longslit", "comoving_rv.db"],
    description="Spectroscopic reduction and radial velocity analysis.",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["astropy", "numpy", "matplotlib", "six", "ccdproc"],
)
