#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="detoxify",
    version="0.0.1",
    description="A python library for detecting toxic comments",
    author="Unitary",
    author_email="laura@unitary.ai",
    url="https://github.com/unitary/detoxify",
    install_requires=[
        "transformers",
        "torch",
    ],
    packages=[
        "detoxify",
    ],
)
