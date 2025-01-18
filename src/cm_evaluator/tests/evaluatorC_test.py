from importlib.resources import files
import json
from typing import List
import networkx as nx

from cm_evaluator.evaluators.evaluatorC import extra_concepts, extra_connections, mislabeled_connections, missing_concepts
from cm_evaluator.graph_utils import create_graph_from_data
from cm_evaluator.models.metrics import MetricConfiguration
from cm_evaluator.models.user_data import MapOutput

dataset_dir = "cm_evaluator.data.datasets.sample_dataset"
reference_file = "reference_solution_en.json"
student_file = "student_solution_minimal_en.json"
language="en"
with open(files(dataset_dir).joinpath(reference_file), 'r', encoding='utf-8') as fd:
        t = json.load(fd)
        cm_master = MapOutput(**t)

with open(files(dataset_dir).joinpath(student_file), 'r', encoding='utf-8') as fd:
    t = json.load(fd)
    cm_student = MapOutput(**t)

with open(files("cm_evaluator.data").joinpath(f'metrics_{language}.json'), 'r', encoding='utf-8') as fd: # type: ignore
    metric_configs: List[MetricConfiguration] = [MetricConfiguration(**data) for data in json.load(fd)]

student_concepts = cm_student.get_node_names()
master_concepts = cm_master.get_node_names()
student_relations = cm_student.convert_to_map_relation()
master_relations = cm_master.convert_to_map_relation()

G_student = create_graph_from_data(student_relations, student_concepts)
G_master = create_graph_from_data(master_relations, master_concepts)


def test_mislabed_conenctions(model):
    graph = nx.DiGraph()
    result = mislabeled_connections(graph, G_master, model, metric_configs)
    assert result.score == 0

    # has at least one edge
    graph.add_edge("AI", "Automation", label="enables")
    result = mislabeled_connections(graph, G_master, model, metric_configs)
    assert result.score == 1.0

    graph.remove_edge("AI", "Automation")
    graph.add_edge("AI", "Automation", label="transforms")
    result = mislabeled_connections(graph, G_master, model, metric_configs)
    assert result.score == 0.94

    result = mislabeled_connections(G_student, G_master, model, metric_configs)
    assert result.score == 1.0

def test_extra_connections(model):
    graph = nx.DiGraph()
    result = extra_connections(graph, G_master, model, metric_configs)
    assert result.score == 0

    graph.add_edge("AI", "Automation", label="enables")
    result = extra_connections(graph, G_master, model, metric_configs)
    assert result.score == 1.0

    graph.add_edge("AI", "Graphics", label="transforms")
    result = extra_connections(graph, G_master, model, metric_configs)
    # one from two edges is extra
    assert result.score == 0.5

    result = extra_connections(G_student, G_master, model, metric_configs)
    assert result.score == 1.0


def test_missing_concepts(model):
    result = missing_concepts([], master_concepts, G_master, model, metric_configs)
    assert result.score == 0

    result = missing_concepts(["Bla"], master_concepts, G_master, model, metric_configs)
    assert result.score == 0

    result = missing_concepts(["AI"], master_concepts,  G_master, model, metric_configs)
    assert result.score == 0.06

    result = missing_concepts(student_concepts, master_concepts, G_master, model, metric_configs)
    assert result.score == 0.24


def test_extra_concepts(model):
    # student has no concepts -> penalize score
    result = extra_concepts([], master_concepts, model, metric_configs)
    assert result.score == 0

    # concept exists, so not an extra
    result = extra_concepts(["AI"], master_concepts, model, metric_configs)
    assert result.score == 1.0

    # one concept in total, one is extra
    result = extra_concepts(["Bla"], master_concepts, model, metric_configs)
    assert result.score == 0