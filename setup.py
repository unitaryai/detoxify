#!/usr/bin/env python
import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir=_PATH_ROOT, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt")) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [ln[: ln.index(comment_char)] if comment_char in ln else ln for ln in lines]
    reqs = [ln for ln in reqs if "-r " not in ln]
    return reqs


def _load_description(path_dir=_PATH_ROOT, readme_file="README.md"):
    with open(os.path.join(path_dir, readme_file)) as fh:
        desc = fh.read()
    return desc


setup(
    name="detoxify",
    version="0.5.1",
    description="A python library for detecting toxic comments",
    long_description=_load_description(),
    long_description_content_type="text/markdown",
    author="Unitary",
    author_email="laura@unitary.ai",
    url="https://github.com/unitaryai/detoxify",
    install_requires=_load_requirements(),
    extras_require=dict(train=_load_requirements("requirements-train.txt")),
    packages=find_packages(include=["detoxify"], exclude=["tests", "src"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
