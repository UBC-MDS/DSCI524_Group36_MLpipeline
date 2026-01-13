"""
Method for comparing fitted scikit-learn models using
a common evaluation metric.
"""

def model_comparison(models, X, y, metric="accuracy",greater_is_better=False):
    """
    Compare multiple fitted scikit-learn models and return the best-performing one.

    Models are evaluated on the same dataset using a user-specified
    evaluation metric. The model with the highest score is returned.

    Parameters
    ----------
    models : list of sklearn.base.BaseEstimator
        A list of fitted scikit-learn model objects that implement
        the ``predict`` method.
    X : pandas.DataFrame or array-like
        Feature matrix used for evaluation.
    y : pandas.Series or array-like
        True target values.
    metric : str, default="accuracy"
        Evaluation metric used for comparison. Must be a valid
        scikit-learn classification metric (e.g. "accuracy",
        "f1", "precision", "recall").

    Returns
    -------
    sklearn.base.BaseEstimator
        The model with the best performance according to the
        selected evaluation metric.

    Raises
    ------
    ValueError
        If the metric is not supported or if models is empty or not a valid sklearn object.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> models = [LogisticRegression().fit(X, y),
    ...           DecisionTreeClassifier().fit(X, y)]
    >>> best_model = model_comparison(models, X, y, metric="accuracy")
    """
    from sklearn import metrics
    from sklearn.base import is_classifier
    import numpy as np
    import warnings

    if not hasattr(metrics, metric):
        raise ValueError(f"{metric} is not a valid sklearn metric")
    
    metric_fn = getattr(metrics, metric)

    valid_models = []
    scores = []

    for model in models:
        if not is_classifier(model):
            warnings.warn(
                f"Skipped model {model.__class__.__name__}: not a classifier",
                UserWarning
            )
            continue
        else:   
            pred = model.predict(X)
            score = metric_fn(y,pred)
            valid_models.append(model)
            scores.append(score)
    
    if not scores:
        raise ValueError("No valid Classification models provided")
    
    if greater_is_better:
        best_idx = np.argmax(scores)
    else:
        best_idx = np.argmin(scores)

    return models[best_idx]