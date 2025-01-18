from importlib.resources import files
import json
from typing import List
import networkx as nx

from cm_evaluator.evaluators.common.common_eval_functions import get_cycles, get_density_difference, isolates, missing_concepts
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


def test_isolates():
    graph = nx.DiGraph()
    
    result = isolates(graph, metric_configs)
    assert result.score == 0
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edges_from([("A", "B"), ("B", "C")])

    result = isolates(graph, metric_configs)
    assert result.score == 0.75
    assert result.isolated_concepts == ["D"]

    graph.add_edges_from([("C", "D")])
    result = isolates(graph, metric_configs)
    assert result.score == 1
    assert result.isolated_concepts == []

def test_get_cycles():
    graph = nx.DiGraph()
    
    result = get_cycles(graph, G_master, metric_configs)
    assert result.score == 0
    
    
    graph.add_edges_from([("A", "B"), ("B", "A")])
    result = get_cycles(graph, G_master, metric_configs)
    assert result.score == 0.75
    assert len(result.cycles_student) == 1
    assert result.cycles_master == []


def test_get_density_difference():
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edges_from([("A", "B"), ("B", "C")])

    m_graph = nx.DiGraph()
    m_graph.add_nodes_from(["A", "B", "C", "D"])
    m_graph.add_edges_from([("A", "B"), ("B", "C")])

    # should show no difference
    result = get_density_difference(graph, m_graph, metric_configs)
    assert result.score == 1.0