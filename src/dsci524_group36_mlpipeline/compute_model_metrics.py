def compute_model_metrics(model, X, y, metrics):
    """
    Compute evaluation metrics for a single fitted machine learning model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A fitted scikit-learn model used for prediction.
    X : array-like
        Feature matrix for evaluation.
    y : array-like
        True target values.
    metrics : dict
        Dictionary where keys are metric names and values are callable metric
        functions from sklearn.metrics.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing one row of evaluation metrics
        corresponding to the input model.

    Raises
    ------
    ValueError
        If the model is not fitted or if metric functions are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import accuracy_score
    >>> from dsci524_group36_mlpipeline.compute_model_metrics import compute_model_metrics
    >>>
    >>> # Load a small example dataset
    >>> df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    >>> X = df.drop(columns=["species"])
    >>> y = df["species"]
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    >>> model = LogisticRegression(max_iter=200).fit(X_train, y_train)
    >>>
    >>> metrics = {"accuracy": accuracy_score}
    >>> compute_model_metrics(model, X_test, y_test, metrics)
       accuracy
    0      ...
    """
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted


    if not hasattr(model, "predict"):
        raise TypeError("model must implement a predict method")

    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("metrics must be a non-empty dictionary")

    for name, fn in metrics.items():
        if not callable(fn):
            raise ValueError(f"Metric '{name}' is not callable")


    try:
        check_is_fitted(model)
    except Exception as e:
        raise ValueError("model must be fitted before evaluation") from e


    y_pred = model.predict(X)
    results = {}
    for name, fn in metrics.items():
        try:
            results[name] = fn(y, y_pred)
        except Exception as e:
            raise ValueError(f"Error computing metric '{name}'") from e


    return pd.DataFrame([results])
