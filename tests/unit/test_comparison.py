"""
A test module that tests the model comparison function.

This test script provides multiple tests to ensure that the model comparison function is performing as expected.
"""

from dsci524_group36_mlpipeline.model_comparison import model_comparison

def test_metric_type():
    """
    Tests that the function returns an error if the wrong data type
    is passed for metric.
    """
    pass

def test_valid_skmetric():
    """
    Tests that the function raises an error if a non sklearn metric
    is passed
    """
    pass

def test_warning_non_classifier():
    """
    Tests that the function raises a warning if a non classification model
    is added to the metric list
    """
    pass

def test_function_output():
    """
    Tests that the output of the function is a valid sklearn classification model
    """
    pass

def test_no_classifier_found():
    """
    Tests that the function raises a value error if no valid classification models are
    passed into the function.
    """
    pass
