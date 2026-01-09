# Welcome to DSCI524_Group36_MLpipeline

|        |        |
|--------|--------|
| Package | [![Latest PyPI Version](https://img.shields.io/pypi/v/dsci524_group36_mlpipeline.svg)](https://pypi.org/project/dsci524_group36_mlpipeline/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/dsci524_group36_mlpipeline.svg)](https://pypi.org/project/dsci524_group36_mlpipeline/)  |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) |


DSCI524_Group36_MLpipeline is a project that simplifies some of the steps in the machine learning process and provides a model comparison function. These methods cover simple tasks in exploratory data analysis (EDA), pipeline creation, computation of metrics and model comparison. This serves ML practitioners with a quick fix to most tasks they may need to perform in the development process without having to import multiple packages.

## Methods available in this package (Function Documentation)

2. Model Pipeline: Create model pipeline for model of choice (logistic regression, SVC or
    random forest) with standardisation for numerical features and one-hot
    encoding for categorical features.
3. EDA functions: Exploratory data analysis of the target column with summary statistics and class distribution visualization
4. Model Metrics Computation: Computes evaluation metrics for a single fitted scikit-learn model using user-specified metric functions and returns the results in a tabular format for reporting or comparison.

## Contributors
- Yi-Ling Chin
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

- Copyright © 2026 Bright, Charlene, Claudia, Tiffany.
- Free software distributed under the [MIT License](./LICENSE).
