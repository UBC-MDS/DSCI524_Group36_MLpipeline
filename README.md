# Welcome to DSCI524_Group36_MLpipeline

|        |        |
|--------|--------|
| Package | [![Latest PyPI Version](https://img.shields.io/pypi/v/dsci524_group36_mlpipeline.svg)](https://pypi.org/project/dsci524_group36_mlpipeline/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/dsci524_group36_mlpipeline.svg)](https://pypi.org/project/dsci524_group36_mlpipeline/)  |
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

## Copyright
- The template for the repository is from [link](https://www.pyopensci.org/python-package-guide/tutorials/create-python-package.html#step-1-set-up-the-package-directory-structure)
- Copyright © 2026 Bright, Charlene, Claudia, Tiffany.
- Free software distributed under the [MIT License](./LICENSE).
