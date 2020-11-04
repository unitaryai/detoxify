#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="detoxify",
    version="0.0.0",
    description="A python library for detecting toxic comments",
    author="Unitary",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/laurahanu/detoxify",
    install_requires=["pytorch-lightning",
                      "transformers",
                      "datasets",
                      "pandas",
                      "kaggle",
                      "scikit-learn",
                      "tqdm"],
    packages=find_packages(),
)
