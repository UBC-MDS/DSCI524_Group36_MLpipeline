"""
A test module that tests the model comparison function.

This test script provides multiple tests to ensure that the model comparison function is performing as expected.
"""
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pytest
from sklearn.base import is_classifier

log = LogisticRegression()
lr = LinearRegression()
rf = RandomForestClassifier()
et = ExtraTreesClassifier() #Will not be fitted


load_data = load_iris()
X = load_data["data"]
y = load_data["target"]

X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2)

log.fit(X_train,y_train)
lr.fit(X_train,y_train)
rf.fit(X_train,y_train)



from dsci524_group36_mlpipeline.model_comparison import model_comparison

def test_valid_skmetric():
    """
    Tests that the function returns an error if the wrong data type
    is passed for metric.
    """
    metric = 4
    best = model_comparison([log, rf], X_train, y_train, metric=metric,greater_is_better=False)

    with pytest.raises(TypeError, match="Expected metric input to be str"): 
        model_comparison([log, rf], X_train, y_train, metric=metric,greater_is_better=False)

def test_warning_non_classifier():
    """
    Tests that the function raises a warning if a non classification model
    is added to the metric list
    """
    metric = "mean_squared_error"
    
    with pytest.warns(UserWarning):
        model_comparison([log, lr], X_train, y_train, metric=metric,greater_is_better=False)

def test_function_output():
    """
    Tests that the output of the function is a valid sklearn classification model
    """
    metric = "mean_squared_error"
    
    best = model_comparison([log, rf], X_train, y_train, metric=metric,greater_is_better=False)
    
    assert is_classifier(best) == True

        

def test_no_classifier_found():
    """
    Tests that the function raises a value error if no valid classification models are
    passed into the function.
    """
    metric = "mean_squared_error"

    with pytest.raises(ValueError, match="No valid Classification models provided"): 
        model_comparison([lr], X_train, y_train, metric=metric,greater_is_better=False)

def test_model_is_fitted():
    """
    Tests if the function raises an error if the passed model has not been fitted
    """
    metric = "mean_squared_error"
    
    with pytest.warns(UserWarning):
        model_comparison([log, et], X_train, y_train, metric=metric,greater_is_better=False)