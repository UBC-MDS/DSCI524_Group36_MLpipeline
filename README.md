# Welcome to DSCI524_Group36_MLpipeline


|        |        |
|--------|--------|
| CI/CD  | [![CI](https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/UBC-MDS/DSCI524_Group36_MLpipeline/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/DSCI524_Group36_MLpipeline) |
| Package | [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) |


DSCI524_Group36_MLpipeline is a project that simplifies some of the steps in the machine learning process and provides a model comparison function. These methods cover simple tasks in exploratory data analysis (EDA), pipeline creation, computation of metrics and model comparison. This package helps machine learning practitioners streamline common workflow steps without needing to rely on multiple external libraries.

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

## Installation

### Install from TestPyPI 

```bash
pip install -i https://test.pypi.org/simple/ dsci524_group36_mlpipeline
```

### Install from PyPI

```bash
pip install dsci524_group36_mlpipeline
```

## Quick Start Example

Below is a minimal, runnable example showing how to train a model and use functions from our package.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from dsci524_group36_mlpipeline.compute_model_metrics import compute_model_metrics, model_comparison, eda

# Load example dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

X = df.drop(columns=["species"])
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# Train model
log = LogisticRegression(max_iter=200)
log.fit(X_train, y_train)

# Define metrics
metrics = {
    "accuracy": accuracy_score,
    "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
}

# Compute metrics
results = compute_model_metrics(log, X_test, y_test, metrics)
print(results)


# Using Model Comparison Function
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
best_model = model_comparison([log, dt], x, y, metric)


# Using eda function
stats, ax = eda(df, "sepal_length")
print("Summary Statistics:")
print(stats)

# Customize the plot using the returned matplotlib Axes object
ax.set_title("Distribution of Sepal Length")
plt.show()
```

## Development setup
To set up the development environment, clone the repository and create the conda environment using the provided environment.yml file:
```bash
git clone https://github.com/UBC-MDS/DSCI524_Group36_MLpipeline.git
cd DSCI524_Group36_MLpipeline
conda env create -f environment.yml
conda activate 524
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
