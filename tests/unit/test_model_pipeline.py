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


@pytest.fixture(params=['lr', 'svc', 'rf'])
def pipeline(request, X):
    output = create_model_pipeline(
        X=X,
        numerical_feat=['age'],
        categorical_feat=['sex'],
        model=request.param
    )
    return output


def test_create_model_pipeline(pipeline):
    # Expected use case
    assert isinstance(pipeline, Pipeline)


def test_fit_model_pipeline(pipeline, X, y):
    pipeline.fit(X, y)
    check_is_fitted(pipeline)
