"""
test_compute_model_metrics.py: test module for compute_model_metrics
"""

import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris

from dsci524_group36_mlpipeline.compute_model_metrics import compute_model_metrics


@pytest.fixture # suggested by chatgpt
def fitted_model_data():
    """
    Fixture providing a fitted sklearn model and evaluation data.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200).fit(X, y)
    return model, X, y


def test_compute_model_metrics_type_error():
    """
    Test if the function raises a TypeError when the model input is invalid.
    """
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    metrics = {"accuracy": accuracy_score}

    with pytest.raises(TypeError, match="predict"):
        compute_model_metrics("not_a_model", X, y, metrics)


def test_compute_model_metrics_value_error_empty_metrics(fitted_model_data):
    """
    Test if the function raises a ValueError when metrics is an empty dictionary.
    """
    model, X, y = fitted_model_data

    with pytest.raises(ValueError, match="non-empty"): # suggested by chatgpt
        compute_model_metrics(model, X, y, metrics={})


def test_compute_model_metrics_invalid_metric_function(fitted_model_data):
    """
    Test if the function raises a ValueError when a metric is not callable.
    """
    model, X, y = fitted_model_data
    metrics = {"accuracy": "not_a_function"}

    with pytest.raises(ValueError, match="callable"):
        compute_model_metrics(model, X, y, metrics)


def test_compute_model_metrics_unfitted_model():
    """
    Test if the function raises a ValueError when the model is not fitted.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression()
    metrics = {"accuracy": accuracy_score}

    with pytest.raises(ValueError, match="fitted"):
        compute_model_metrics(model, X, y, metrics)


def test_compute_model_metrics_valid_input(fitted_model_data):
    """
    Test if the function returns a DataFrame when given valid inputs.
    """
    model, X, y = fitted_model_data
    metrics = {"accuracy": accuracy_score}

    result = compute_model_metrics(model, X, y, metrics)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1


def test_compute_model_metrics_multiple_metrics(fitted_model_data):
    """
    Test if the function correctly computes multiple metrics.
    """
    model, X, y = fitted_model_data
    metrics = {
        "accuracy": accuracy_score,
        "f1": partial(f1_score, average="macro"),
    }

    result = compute_model_metrics(model, X, y, metrics)

    assert "accuracy" in result.columns
    assert "f1" in result.columns

# Additional tests suggested by chatgpt
def test_compute_model_metrics_values_are_numeric(fitted_model_data):
    """
    Test that computed metric values are numeric.
    """
    model, X, y = fitted_model_data
    metrics = {"accuracy": accuracy_score}

    result = compute_model_metrics(model, X, y, metrics)

    assert isinstance(result.iloc[0]["accuracy"], (int, float))
