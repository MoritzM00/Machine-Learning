# Intro to Machine Learning

![Tests](https://github.com/MoritzM00/Machine-Learning/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/github/license/MoritzM00/Machine-Learning?color=blue)

---

## Overview
This project implements several well known statistical methods used to analyse high-dimensional data.
The algorithms do not aim to be very efficient, but to foster the understanding of the methods.

## Table of contents
- Linear Models (such as Linear Regression, Ridge Regression etc.)
- Decomposition (Principal Component Analysis)
- Neural Networks (later)

## Installation

This project uses [Poetry](https://python-poetry.org/ "python-poetry.org") as its dependency manager.
If you do not already have it, install Poetry.
For instructions, follow this link:

https://github.com/python-poetry/poetry/blob/master/README.md#installation


Once you have Poetry installed, clone the repository and execute:

```bash
poetry config virtualenvs.in-project true
poetry install
```

This creates a virtual environment and installs the required packages.
