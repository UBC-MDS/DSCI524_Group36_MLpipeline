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
    >>> metrics = {"accuracy": accuracy_score, "f1": f1_score}
    >>> compute_model_metrics(model, X_test, y_test, metrics)
       accuracy    f1
    0     0.88    0.86
    """
    pass
