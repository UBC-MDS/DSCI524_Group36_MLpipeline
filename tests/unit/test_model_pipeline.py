from dsci524_group36_mlpipeline.model_pipeline import create_model_pipeline
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import pytest

# Prepare test data
data = fetch_openml(name='titanic', version=1).frame
# Select X columns and remove rows with missing data
data = data[['pclass', 'survived', 'sex', 'age']].dropna()


@pytest.fixture()
def X():
    return data.drop(columns='survived')


@pytest.fixture()
def y():
    return data.survived


# Expected use case
@pytest.fixture(params=['lr', 'svc', 'rf'])
def pipeline(request, X):
    output = create_model_pipeline(
        X=X,
        numerical_feat=['age'],
        categorical_feat=['sex'],
        model=request.param
    )
    return output


# Numerical categories not specified
@pytest.fixture(params=['lr', 'svc', 'rf'])
def pipeline_empty_list(request, X):
    output = create_model_pipeline(
        X=X,
        categorical_feat=['sex'],
        model=request.param
    )
    return output


def test_create_model_pipeline(pipeline, pipeline_empty_list):
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline_empty_list, Pipeline)


def test_fit_model_pipeline(pipeline, pipeline_empty_list, X, y):
    pipeline.fit(X, y)
    check_is_fitted(pipeline)

    pipeline_empty_list.fit(X, y)
    check_is_fitted(pipeline_empty_list)
