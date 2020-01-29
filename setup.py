#!/usr/bin/env python

from setuptools import setup
setup(
    name="strange",
    packages=["strange"],
    version="0.1",
    author="Deren Eaton, Patrick McKenzie, and Jianjun Jin",
    author_email="de2356@columbia.edu",
    install_requires=[
        "ipcoal",
        "pandas>=0.16",
        "toytree",
        "toyplot",
    ],
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
)
