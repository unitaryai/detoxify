#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="detoxify",
    version="0.1.0",
    description="A python library for detecting toxic comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Unitary",
    author_email="laura@unitary.ai",
    url="https://github.com/unitaryai/detoxify",
    install_requires=[
        "transformers",
        "torch",
    ],
    packages=find_packages(exclude=("tests", "src")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
