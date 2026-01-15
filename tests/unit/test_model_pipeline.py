from dsci524_group36_mlpipeline.model_pipeline import create_model_pipeline
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
import pytest

# Prepare test data
data = fetch_openml(name='titanic', version=1).frame
# Select X columns and remove rows with missing data
X = data[['pclass', 'sex', 'age']].dropna()


@pytest.mark.parametrize("model", ['lr', 'svc', 'rf'])
def test_create_model_pipeline(model):
    # Expected use case
    output = create_model_pipeline(
        X=X,
        numerical_feat=['age'],
        categorical_feat=['sex'],
        model=model
    )
    assert isinstance(output, Pipeline)
