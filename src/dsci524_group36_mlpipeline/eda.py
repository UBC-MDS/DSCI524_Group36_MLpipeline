"""
Exploratory data analysis of a target column including 
summary statistics and distribution visualization.
"""

def eda(X, y):
    """
    Perform exploratory data analysis on a single numeric column of a DataFrame.

    This function computes descriptive statistics for a specified column and
    generates a histogram to visualize its distribution. It is written
    defensively and will raise informative errors when invalid inputs or
    unsupported data types are provided.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing the column to be analyzed. Typically the target column
    y : str
        Name of the column in `X` for which summary statistics and a histogram
        will be generated. The column must exist in `X` and contain numeric
        values.

    Returns
    -------
    summary_stats : pandas.Series
        Descriptive statistics for column `y`, as returned by
        pandas.Series.describe.
    histogram : matplotlib.axes.Axes
        Matplotlib Axes object containing the histogram of column `y`.

    Raises
    ------
    TypeError
        If `X` is not a pandas DataFrame.
        If `y` is not a string.
        If column `y` is not numeric.
    KeyError
        If column `y` does not exist in `X`.
    ValueError
        If column `y` is empty.
        If column `y` contains only missing values (NaNs).

    Notes
    -----
    This function creates a matplotlib plot but does not display it. To render
    the histogram, call `matplotlib.pyplot.show()` after invoking this
    function.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas.api.types import is_numeric_dtype 

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Input X has type {type(X)}, must be a pandas DataFrame")

    if not isinstance(y, str):
        raise TypeError(f"Target column name y has type {type(y)}, must be a string")

    if y not in X.columns:
        raise KeyError(f"Column '{y}' not found in DataFrame. Columns in the data: {list(X.columns)}")

    if X[y].empty:
        raise ValueError(f"Column '{y}' is empty (length=0) and can't be analyzed")

    if X[y].dropna().empty:
        raise ValueError(f"Column '{y}' contains only missing values (NaNs)")

    if not is_numeric_dtype(X[y]):
        raise TypeError(f"Column '{y}' has datatype {X[y].dtype} but must be numeric to generate a histogram")


    summary_stats = X[y].describe()
    histogram = X[y].plot(kind="hist")

    return summary_stats, histogram
