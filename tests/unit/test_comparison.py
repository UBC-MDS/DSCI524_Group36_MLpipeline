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


@pytest.fixture #suggestion from claude
def fitted_models():
    """Fixture to provide fitted models for testing"""
    load_data = load_iris()
    X = load_data["data"]
    y = load_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log = LogisticRegression()
    rf = RandomForestClassifier()
    lr = LinearRegression()
    et = ExtraTreesClassifier() 
    
    log.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    return {
        'log': log,
        'rf': rf,
        'lr': lr,
        'et': et,
        'X_train': X_train,
        'y_train': y_train
    }


from dsci524_group36_mlpipeline.model_comparison import model_comparison

def test_valid_skmetric():
    """
    Tests that the function returns an error if the wrong data type
    is passed for metric.
    """
    metric = 4

    with pytest.raises(TypeError, match="Expected metric input to be str"): 
        model_comparison([fitted_models['log'], fitted_models['rf']], 
                        fitted_models['X_train'], fitted_models['y_train'], 
                        metric=metric, greater_is_better=False)

def test_warning_non_classifier():
    """
    Tests that the function raises a warning if a non classification model
    is added to the metric list
    """
    metric = "mean_squared_error"
    
    with pytest.warns(UserWarning):
        model_comparison([fitted_models['log'], fitted_models['lr']], fitted_models['X_train'], fitted_models['y_train'], metric=metric,greater_is_better=False)

def test_function_output():
    """
    Tests that the output of the function is a valid sklearn classification model
    """
    metric = "mean_squared_error"
    
    best = model_comparison([fitted_models['log'], fitted_models['rf']], fitted_models['X_train'], fitted_models['y_train'], metric=metric,greater_is_better=False)
    
    assert is_classifier(best) 

        

def test_no_classifier_found():
    """
    Tests that the function raises a value error if no valid classification models are
    passed into the function.
    """
    metric = "mean_squared_error"

    with pytest.warns(UserWarning):  # Expect the warning about non-classifier
        with pytest.raises(ValueError, match="No valid Classification models provided"): 
            model_comparison([fitted_models['lr']], fitted_models['X_train'], fitted_models['y_train'], metric=metric, greater_is_better=False)

def test_model_is_fitted():
    """
    Tests if the function raises an error if the passed model has not been fitted
    """
    metric = "mean_squared_error"
    
    with pytest.warns(UserWarning):
        model_comparison([fitted_models['log'], fitted_models['et']], fitted_models['X_train'], fitted_models['y_train'], metric=metric,greater_is_better=False)



# Additional tests suggested by Claude AI sonnet 4.5

def test_empty_model_list():
    """
    Tests that the function raises a ValueError when an empty list of models
    is provided.
    """
    metric = "accuracy"
    
    with pytest.raises(ValueError, match="No valid Classification models provided"):
        model_comparison([], fitted_models['X_train'], fitted_models['y_train'], metric=metric, greater_is_better=True)


def test_unknown_metric_name():
    """
    Tests that the function raises a ValueError when an invalid/unknown
    sklearn metric name is provided.
    """
    metric = "this_metric_does_not_exist"
    
    with pytest.raises(ValueError, match="this_metric_does_not_exist is not a valid sklearn metric"):
        model_comparison([fitted_models['log'], fitted_models['rf']], fitted_models['X_train'], fitted_models['y_train'], metric=metric, greater_is_better=True)


def test_all_models_unfitted():
    """
    Tests that the function raises a ValueError when all provided models
    are unfitted.
    """
    # Create unfitted models
    unfitted_log = LogisticRegression()
    unfitted_rf = RandomForestClassifier()
    
    metric = "accuracy"
    
    with pytest.warns(UserWarning):  # Expect warnings about unfitted models
        with pytest.raises(ValueError, match="No valid Classification models provided"):
            model_comparison([unfitted_log, unfitted_rf], fitted_models['X_train'], fitted_models['y_train'], 
                           metric=metric, greater_is_better=True)


def test_invalid_x_train_shape():
    """
    Tests that the function raises an error when X_train has incompatible
    shape with the fitted models (wrong number of features).
    """
    metric = "accuracy"
    
    # Create X with wrong number of features (iris has 4 features, we'll use 2)
    invalid_X = fitted_models['X_train'][:, :2]
    
    with pytest.raises(ValueError):
        model_comparison([fitted_models['log'], fitted_models['rf']], invalid_X, fitted_models['y_train'], metric=metric, greater_is_better=True)


def test_invalid_y_train_shape():
    """
    Tests that the function raises an error when y_train has incompatible
    shape with X_train (mismatched number of samples).
    """
    metric = "accuracy"
    
    # Create y with wrong number of samples
    invalid_y = fitted_models['y_train'][:len(fitted_models['y_train'])//2]
    
    with pytest.raises(ValueError):
        model_comparison([fitted_models['log'], fitted_models['rf']], fitted_models['X_train'], invalid_y, metric=metric, greater_is_better=True)


def test_invalid_x_train_type():
    """
    Tests that the function handles invalid X_train data types appropriately.
    """
    metric = "accuracy"
    
    # Pass a string instead of array-like
    invalid_X = "not an array"
    
    with pytest.raises((ValueError, TypeError, AttributeError)):
        model_comparison([fitted_models['log'], fitted_models['rf']], invalid_X, fitted_models['y_train'], metric=metric, greater_is_better=True)