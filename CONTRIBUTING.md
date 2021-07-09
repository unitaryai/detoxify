

# Contributor's Guide

How to make contributions to this repo.

First fork the repo, and clone your copy onto your local machine, and navigate there from the command line.

Activate a virtual environment. Then install package dependencies:

```sh
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

Run tests (may take 90 seconds):

```sh
pytest tests --disable-pytest-warnings
```
