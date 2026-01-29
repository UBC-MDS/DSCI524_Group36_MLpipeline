# Welcome to DSCI524_Group36_MLpipeline


|        |        |
|--------|--------|
| CI/CD  | [![CI](https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/UBC-MDS/DSCI524_Group36_MLpipeline/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/DSCI524_Group36_MLpipeline) |
| Package | [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) |


DSCI524_Group36_MLpipeline is a project that simplifies some of the steps in the machine learning process and provides a model comparison function. These methods cover simple tasks in exploratory data analysis (EDA), pipeline creation, computation of metrics and model comparison. This serves ML practitioners with a quick fix to most tasks they may need to perform in the development process without having to import multiple packages.

## Methods available in this package (Function Documentation)

1. EDA functions: Exploratory data analysis of the target column with summary statistics and class distribution visualization
2. Model Pipeline: Create model pipeline for model of choice (logistic regression, SVC or
    random forest) with standardisation for numerical features and one-hot
    encoding for categorical features.
3. Model Metrics Computation: Computes evaluation metrics for a single fitted scikit-learn model using user-specified metric functions and returns the results in a tabular format for reporting or comparison.
4. Model Comparison: Accepts a list of scikit learn model objects, a dataframe of observations, a series of actual values and a comparison metric to be used, and returns a single model.

What makes our package different from others is that it combines multiple steps in the machine learning process into one package, making it easier for users to perform these tasks without having to switch between different libraries. It also provides a simple and consistent interface for users to interact with, making it more user-friendly. Other packages like PyCaret are meant for complex ML tasks and may have a steeper learning curve for beginners. Our package is designed to be simple and easy to use, making it accessible to a wider range of users.

## Contributors
- Charlene Chin
- Claudia Liauw
- Bright Arafat Bello
- Tiffany Chu

## Get started

You can install this package into your preferred Python environment using pip:

```bash
$ pip install dsci524_group36_mlpipeline
```

To use dsci524_group36_mlpipeline in your code:

```python
>>> import dsci524_group36_mlpipeline
>>> dsci524_group36_mlpipeline.model_comparison(objects, x, y, metric)
```

## Development setup
To set up the development environment, clone the repository and create the conda environment using the provided environment.yml file:
```bash
git clone https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline.git
cd DSCI524_Group36_MLpipeline
conda env create -f environment.yml
conda activate dsci524_group36_mlpipeline
```

## Install package (development mode)
Install the package in editable mode with testing dependencies:
```bash
pip install -e ".[tests]"
```
This allows developers to make changes to the source code while using the package.

## Run tests
To run the unit tests locally, use:
```bash
pytest
```
Test coverage can be checked using:
```bash
pytest --cov
```

## Build documentation
Package documentation is built using quartodoc and Quarto.

To build the documentation locally:
```bash
quarto render docs
```
The rendered documentation will be available in the docs/_site directory.

## Deploy documentation (automated)
Documentation is automatically deployed to GitHub Pages using a GitHub Actions workflow whenever changes are pushed to the main branch.
No manual deployment steps are required.

## Documentation
The full package documentation, including function references and usage examples, is available at:
https://ubc-mds.github.io/DSCI524_Group36_MLpipeline/reference/

## Copyright
- The template for the repository is from [link](https://www.pyopensci.org/python-package-guide/tutorials/create-python-package.html#step-1-set-up-the-package-directory-structure)
- Copyright © 2026 Bright, Charlene, Claudia, Tiffany.
- Free software distributed under the [MIT License](./LICENSE).
