from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd


def create_model_pipeline(X, numerical_feat=[], categorical_feat=[], model='lr'):
    """
    Create model pipeline for model of choice (logistic regression, SVC or
    random forest) with standardisation for numerical features and one-hot
    encoding for categorical features. Any remaining features are passed
    through with no preprocessing.

    Parameters
    ----------
    X : pd.DataFrame
        Data without target column. Should not contain missing values.
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
    TypeError
        If input types are wrong.
    ValueError
        If model not in specified list or columns are not found in dataframe.

    Examples
    --------
    >>> pipeline = create_model_pipeline(X, ['age'], ['sex'], 'lr')
    >>> pipeline.fit(X, y)
    >>> predictions = pipeline.predict(X)
    """
    # Input checks
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected X to be of type pd.DataFrame, got {type(X)}")

    if not isinstance(numerical_feat, list):
        raise TypeError(f"Expected numerical_feat to be of type list, got {type(numerical_feat)}")

    if not isinstance(categorical_feat, list):
        raise TypeError(f"Expected categorical_feat to be of type list, got {type(categorical_feat)}")

    if model not in ['lr', 'svc', 'rf']:
        raise ValueError(f"Expected model to be 'lr', 'svc' or 'rf, got {model}")
    
    if X.isna().any().any():
        raise ValueError("Dataframe contains missing values.")

    missing_cols = (
        [feat for feat in numerical_feat if feat not in X.columns]
        + [feat for feat in categorical_feat if feat not in X.columns]
    )
    if len(missing_cols) != 0:
        raise ValueError(f"{missing_cols} not found in dataframe")

    # Build pipeline
    model_dict = {
        'lr': LogisticRegression(),
        'svc': SVC(),
        'rf': RandomForestClassifier(random_state=123)
    }

    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_feat),
        (OneHotEncoder(drop='if_binary', handle_unknown='ignore'), categorical_feat)
    )

    pipe = make_pipeline(
        preprocessor,
        model_dict[model]
    )

    return pipe
