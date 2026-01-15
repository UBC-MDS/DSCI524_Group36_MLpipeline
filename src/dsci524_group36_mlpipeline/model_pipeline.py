def create_model_pipeline(X, numerical_feat, categorical_feat, model='lr'):
    """
    Create model pipeline for model of choice (logistic regression, SVC or
    random forest) with standardisation for numerical features and one-hot
    encoding for categorical features. Any remaining features are passed
    through with no preprocessing.

    Parameters
    ----------
    X : pd.DataFrame
        Data without target column.
    numerical_feat : list
        Names of numerical columns to be standardised.
    categorical_feat : list
        Names of categorical columns to be one-hot encoded.
    model : {'lr', 'svc', 'rf'}, default = 'lr'
        Model to include in pipeline.
        `lr`: `LogisticRegression`
        `svc`: `SVC`
        `rf`: `RandomForestClassifier`

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline with standardisation, one-hot encoding and specified model.

    Raises
    ------
    ValueError
        If input types are wrong or columns are not found in dataframe.

    Examples
    --------
    >>> pipeline = create_model_pipeline(X, ['age'], ['sex'], 'lr')
    >>> pipeline.fit(X, y)
    >>> predictions = pipeline.predict(X)
    """
    pass
