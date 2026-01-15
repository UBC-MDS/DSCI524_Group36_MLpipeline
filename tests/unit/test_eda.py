"""
test_eda.py: test module for eda function to ensure it is performing as expected
"""

import pandas as pd
import matplotlib.axes
import pytest
from sklearn.datasets import load_iris

import pandas as pd
import matplotlib
import matplotlib.axes
import pytest
from sklearn.datasets import load_iris

from dsci524_group36_mlpipeline.eda import eda

@pytest.fixture
def iris_df():
    """
    fixture that creates a pandas DataFrame using the iris dataset, target column titled 'target'
    """
    iris = load_iris() 
    df = pd.DataFrame(iris.data, columns=iris.feature_names) 
    df["target"] = iris.target                            
    return df



def test_eda_valid_numeric_column(iris_df):
    '''
    Test that eda() function returns correct types and values for numeric columns
    '''
    stats, hist = eda(iris_df, "sepal length (cm)")
    assert isinstance(stats, pd.Series)
    assert stats["count"] == len(iris_df)
    assert isinstance(hist, matplotlib.axes.Axes)

def test_eda_numeric_column_with_missing_values():
    """
    Tests that eda correctly handles numeric columns containing NA values by excluding them from summary statistics
    """
    df = pd.DataFrame({"x": [1, 2, None, 4, 5]})
    stats, hist = eda(df, "x")

    assert stats["count"] == 4
    assert isinstance(hist, matplotlib.axes.Axes)


def test_eda_non_dataframe_input():
    """
    Tests that eda raises a TypeError when the input data is not a pandas DataFrame
    """
    with pytest.raises(TypeError, match="pandas DataFrame"):
        eda([1, 2, 3], "x")

def test_eda_missing_column(iris_df):
    """
    Tests that eda raises a KeyError when the specified column does not exist in the DataFrame
    """
    with pytest.raises(KeyError, match="not found"):   #suggested by ChatGPT5.2 to add match=not found 
        eda(iris_df, "not_a_column")

def test_eda_non_numeric_column():
    """
    Tests that eda raises a TypeError when the specified column is non-numeric and cannot be plotted as a histogram
    """
    df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})

    with pytest.raises(TypeError, match="must be numeric"):
        eda(df, "cat")

def test_eda_all_missing_column():
    """
    Tests that eda raises a ValueError when the specified column contains only missing values
    """
    df = pd.DataFrame({"x": [None, None, None]})

    with pytest.raises(ValueError, match="missing values"):
        eda(df, "x")

def test_eda_single_value_column():   #suggested by ChatGPT5.2 to add this test
    """
    Tests that eda correctly handles a numeric column containing a single repeated value
    """
    df = pd.DataFrame({"x": [10, 10, 10, 10]})

    stats, hist = eda(df, "x")

    assert stats["mean"] == 10
    assert stats["std"] == 0.0
    assert isinstance(hist, matplotlib.axes.Axes)

def test_eda_non_string_column_name(iris_df):   #suggested by ChatGPT5.2 to add this test
    """
    Tests that eda raises a TypeError when the target column name is not a string
    """
    with pytest.raises(TypeError, match="must be a string"):
        eda(iris_df, 123)