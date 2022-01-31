#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="detoxify",
    version="0.4.0",
    description="A python library for detecting toxic comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Unitary",
    author_email="laura@unitary.ai",
    url="https://github.com/unitaryai/detoxify",
    install_requires=[
        "transformers >= 3.2.0",
        "torch >= 1.7.0",
        "sentencepiece >= 0.1.94",
    ],
    packages=find_packages(include=["detoxify"], exclude=["tests", "src"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
