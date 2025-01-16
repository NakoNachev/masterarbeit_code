from importlib.resources import files
import json
from typing import List
from cm_evaluator.models.metrics import MetricConfiguration
import pytest
from sentence_transformers import SentenceTransformer
@pytest.fixture(scope="session")
def model():
    """Initialize the model only once per session."""
    model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
    return model
@pytest.fixture(scope="session")
def metric_configs():
    with open(files("cm_evaluator.data.").joinpath("metrics_en.json"), 'r', encoding='utf-8') as fd:
        metric_configs: List[MetricConfiguration] = [MetricConfiguration(**data) for data in json.load(fd)]
    return metric_configs