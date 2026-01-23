from dsci524_group36_mlpipeline.model_pipeline import create_model_pipeline
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import pytest

# Prepare test data
titanic = fetch_openml(name='titanic', version=1).frame
# Select columns and remove rows with missing data
data = titanic[['pclass', 'survived', 'sex', 'age']].dropna()


@pytest.fixture()
def X():
    "Titanic data without target column."
    return data.drop(columns='survived')


@pytest.fixture()
def y():
    "Titanic data target column."
    return data.survived


# Expected use case
@pytest.fixture(params=['lr', 'svc', 'rf'])
def pipeline(request, X):
    """
    Unfitted pipeline with numeric feature age and categorical feature sex,
    parametrized with model.
    """
    output = create_model_pipeline(
        X=X,
        numerical_feat=['age'],
        categorical_feat=['sex'],
        model=request.param
    )
    return output


def test_create_model_pipeline(pipeline):
    """Test that output of `create_model_pipeline` is a Pipeline object."""
    assert isinstance(pipeline, Pipeline)


def test_fit_model_pipeline(pipeline, X, y):
    """Test that output of `create_model_pipeline` is able to fit to data."""
    pipeline.fit(X, y)
    check_is_fitted(pipeline)


# Edge case: numerical categories not specified
@pytest.fixture(params=['lr', 'svc', 'rf'])
def pipeline_empty_list(request, X):
    """
    Unfitted pipeline with numeric features not specified, parametrized with model.
    """
    output = create_model_pipeline(
        X=X,
        categorical_feat=['sex'],
        model=request.param
    )
    return output


def test_create_model_pipeline_empty_list(pipeline_empty_list, X, y):
    """Test that output of `create_model_pipeline` is a Pipeline object and
    is able to fit to data when numerical categories are not specified."""
    assert isinstance(pipeline_empty_list, Pipeline)
    pipeline_empty_list.fit(X, y)
    check_is_fitted(pipeline_empty_list)


# Edge case: missing values
def test_create_model_pipeline_missing_values():
    "Test that missing values in `X` raises ValueError."
    with pytest.raises(ValueError):
        create_model_pipeline(
            X=titanic,
            numerical_feat=['age'],
            categorical_feat=['sex']
        )


# Edge case: columns do not exist
def test_create_model_pipeline_missing_col(X):
    "Test that specifying columns that don't exist in `X` raises ValueError."
    with pytest.raises(ValueError):
        create_model_pipeline(
            X=X,
            numerical_feat=['age'],
            categorical_feat=['sex', 'embarked']
        )


# Test input types
def test_create_model_pipeline_not_dataframe():
    "Test that non-dataframe input for `X` raises TypeError."
    with pytest.raises(TypeError):
        create_model_pipeline(
            X=[1, 2, 3],
            numerical_feat=['age'],
            categorical_feat=['sex']
        )


def test_create_model_pipeline_num_not_list(X):
    "Test that non-list input for `numerical_feat` raises TypeError."
    with pytest.raises(TypeError):
        create_model_pipeline(
            X=X,
            numerical_feat=1,
            categorical_feat=['sex']
        )


def test_create_model_pipeline_cat_not_list(X):
    "Test that non-list input for `categorical_feat` raises TypeError."
    with pytest.raises(TypeError):
        create_model_pipeline(
            X=X,
            numerical_feat=['age'],
            categorical_feat=1
        )


def test_create_model_pipeline_wrong_model(X):
    "Test that invalid options for `model` raises ValueError."
    with pytest.raises(ValueError):
        create_model_pipeline(
            X=X,
            numerical_feat=['age'],
            categorical_feat=['sex'],
            model='logreg'
        )
