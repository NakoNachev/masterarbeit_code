from importlib.resources import files
import json
from pathlib import Path
from typing import List, Literal

from cm_evaluator.decorators.decorators import time_and_log_decorator
from cm_evaluator.evaluators.evaluator_utils import create_output
from cm_evaluator.models.helper_models import MapRelation
from cm_evaluator.models.user_data import MapOutput
from sentence_transformers import SentenceTransformer
import networkx as nx
from pyvis.network import Network

from cm_evaluator.graph_utils import create_graph_from_data
from cm_evaluator.models.metrics import ExtraConceptsMetric, ExtraConnectionsMetric, Metric, MetricConfiguration, MislabeledConnectionsMetric, MissingConceptsMetric, UnusedLabelsMetric
from cm_evaluator.utilities import get_most_similar_word, get_nodes_importance, words_are_similar, common_words
from cm_evaluator.logging.base_logger import logger

from cm_evaluator.evaluators.common import common_eval_functions

def evaluate(
        cm_student: MapOutput, 
        cm_master: MapOutput, 
        model: SentenceTransformer,
        language: Literal["de", "en"] = "en",
        dump_output: bool = False):
    logger.debug(f'Evaluating concept map of user {cm_student.user}')
    student_concepts = cm_student.get_node_names()
    master_concepts = cm_master.get_node_names()
    student_relations = cm_student.convert_to_map_relation()
    master_relations = cm_master.convert_to_map_relation()

    G_student = create_graph_from_data(student_relations, student_concepts)
    G_master = create_graph_from_data(master_relations, master_concepts)
    metric_configs = []
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_json = output_dir / f'output_{cm_student.user}.json'
    logger.debug("Read up the metrics configuration file")
    with open(files("cm_evaluator.data").joinpath(f'metrics_{language}.json'), 'r', encoding='utf-8') as fd: # type: ignore
        metric_configs: List[MetricConfiguration] = [MetricConfiguration(**data) for data in json.load(fd)]

    metric_function_mapper = {
        "1": {"func_name": "missing_connections", "args": [student_relations, master_relations, model, metric_configs], "common_func": True},
        "2": {"func_name": "mislabeled_connections", "args": [G_student, G_master, model, metric_configs]},
        "3": {"func_name": "extra_connections", "args": [G_student, G_master, model, metric_configs]},
        "4": {"func_name": "missing_concepts", "args": [student_concepts, master_concepts, G_master, model, metric_configs, language]},
        "5": {"func_name": "extra_concepts", "args": [student_concepts, master_concepts, model, metric_configs]},
        "6": {"func_name": "unused_edge_labels", "args": [student_relations, master_relations, model, metric_configs]},
        "7": {"func_name": "isolates", "args": [G_student, metric_configs, ], "common_func": True},
        "8": {"func_name": "topologic_similarity", "args": [student_relations, master_relations, G_student, G_master, metric_configs, model], "common_func": True},
        "9": {"func_name": "get_centrality_similarity", "args": [G_student, G_master, metric_configs, model], "common_func": True},
        "10": {"func_name": "get_density_difference", "args": [G_student, G_master, metric_configs], "common_func": True},
        "11": {"func_name": "get_cycles", "args": [G_student, G_master, metric_configs], "common_func": True},
    }

    output_metrics: List[Metric] = []
    for _, val in metric_function_mapper.items():
        output_metric = call_func(val)
        logger.debug(f'got metric {output_metric.__dict__}')
        if output_metric and isinstance(output_metric, Metric):
            output_metrics.append(output_metric)

    logger.debug("Started dumping metrics output in the output file")
    output = create_output(student_relations, master_relations, output_metrics)
    if dump_output:
        with output_json.open('w', encoding='utf-8') as fd:
            json.dump(output, fd, ensure_ascii=False)
    return output


def call_func(val):
    func_name = val["func_name"]
    args = val["args"]

    # Get the function from globals and call it with the arguments
    func = globals().get(func_name)
    if func and callable(func):
        return func(*args)
    
    # Function is shared between evaluators
    if val.get("common_func", False):
        func = getattr(common_eval_functions, func_name, None)
        if func and callable(func):
            return func(*args)
    else:
        print(f"Function {func_name} not found in current or common scope or not callable")

@time_and_log_decorator
def mislabeled_connections(
        student_map: nx.DiGraph, 
        master_map: nx.DiGraph, 
        model:SentenceTransformer,
        metric_configs: List[MetricConfiguration]):
    # metric id 2
    reference_config = next((config for config in metric_configs if config.id == 2), None)
    mislabeled_connections = []
    student_edges = student_map.edges()
    master_concepts = list(master_map.nodes())

    if reference_config:
        if len(student_edges) == 0:
            return MislabeledConnectionsMetric(
            label=reference_config.label,
            label_key="mislabeled_connections",
            explanation=reference_config.explanation,
            score=0,
            mislabeled_connections=[]
            )

        for edge in student_edges:
            u, v = edge

            node1_candidate = get_most_similar_word(u, master_concepts, model)
            node2_candidate = get_most_similar_word(v, master_concepts, model)

            if master_map.has_edge(node1_candidate, node2_candidate):
                master_label = master_map.get_edge_data(node1_candidate, node2_candidate)['label']
                student_label = student_map.get_edge_data(u, v)['label']

                if master_label != student_label:
                    mislabeled_connections.append(f'{u} {student_label} {v} ({master_label})')

        score = 1 - (len(mislabeled_connections) / len(list(master_map.edges())))

        return MislabeledConnectionsMetric(
            label=reference_config.label,
            label_key="mislabeled_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            mislabeled_connections=mislabeled_connections
        )

@time_and_log_decorator    
def extra_connections(
        student_map: nx.DiGraph, 
        master_map: nx.DiGraph, 
        model:SentenceTransformer,
        metric_configs: List[MetricConfiguration]):
    # metric id 3
    reference_config = next((config for config in metric_configs if config.id == 3), None)
    student_edges = student_map.edges()
    master_concepts = list(master_map.nodes())
    extra_connections = []

    if reference_config:
        if len(student_edges) == 0:

            return ExtraConnectionsMetric(
                label=reference_config.label,
                label_key="extra_connections",
                explanation=reference_config.explanation,
                score=0,
                extra_connections=[f"{edge[0]} {student_map.get_edge_data(edge[0], edge[1])['label']} {edge[1]}" for edge in extra_connections]
            )

        for edge in student_edges:
            u, v = edge
            # find the closest candidate and check if edge between the nodes exists
            node1_candidate = get_most_similar_word(u, master_concepts, model)
            node2_candidate = get_most_similar_word(v, master_concepts, model)
            if not (master_map.has_edge(node1_candidate, node2_candidate) or master_map.has_edge(node2_candidate, node1_candidate)):
                extra_connections.append(edge)
            
        score =  1 - (len(extra_connections) / len(list(student_map.edges()))) if len(list(student_map.edges())) != 0 else 1

        return ExtraConnectionsMetric(
            label=reference_config.label,
            label_key="extra_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            extra_connections=[f"{edge[0]} {student_map.get_edge_data(edge[0], edge[1])['label']} {edge[1]}" for edge in extra_connections]
        )

@time_and_log_decorator
def missing_concepts(
        student_concepts: List[str],
        master_concepts: List[str], 
        graph_master: nx.DiGraph, 
        model: SentenceTransformer,
        metric_configs: List[MetricConfiguration],
        language: Literal["de", "en"] = "en"):
    # metric id 4
    reference_config = next((config for config in metric_configs if config.id == 4), None)
    remaining_master = set(master_concepts) - set(student_concepts)
    similar_concepts = common_words(master_concepts, student_concepts, model)
    remaining_after_cleanup = list(set(remaining_master) - set(similar_concepts))

    if reference_config:

        nodes_importance = get_nodes_importance(graph_master, language)
        score = 1 - (len(remaining_after_cleanup) / len(master_concepts))

        return MissingConceptsMetric(
            label=reference_config.label,
            label_key="missing_concepts",
            explanation=reference_config.explanation,
            score=round(score,2),
            missing_concepts=remaining_after_cleanup,
            concepts_importance=nodes_importance,
            missing_concepts_importance={key: value for key, value in nodes_importance.items() if key in remaining_after_cleanup}
        )

@time_and_log_decorator
def extra_concepts(
        student_concepts: List[str], 
        master_concepts: List[str], 
        model: SentenceTransformer,
        metric_configs: List[MetricConfiguration]):
    # metric id 5

    # identify which concepts from the student have no 1:1 similarity with the master concepts
    remaining_student = set(student_concepts) - set(master_concepts)
    similar_concepts = []
    reference_config = next((config for config in metric_configs if config.id == 5), None)

    if reference_config:

        if len(student_concepts) == 0:
            return ExtraConceptsMetric(
                label=reference_config.label,
                label_key="extra_concepts",
                explanation=reference_config.explanation,
                score=0,
                extra_concepts=[]
            )

        # calculate for each of the remaining concepts the similarity score to the master concepts
        # if there is a candidate with sim. >= threshold -> remove it from the list
        for word in list(remaining_student):
            similar = list(filter(lambda w_master: words_are_similar(w_master, word, model), master_concepts))
            if similar:
                similar_concepts.append(word)
        
        remaining_after_cleanup = list(set(remaining_student) - set(similar_concepts))
        score =  1 - (len(remaining_after_cleanup) / len(student_concepts)) if len(remaining_after_cleanup) > 0 else 1

        return ExtraConceptsMetric(
            label=reference_config.label,
            label_key="extra_concepts",
            explanation=reference_config.explanation,
            score=round(score,2),
            extra_concepts=remaining_after_cleanup
        )

@time_and_log_decorator
def unused_edge_labels(
        relations_student: List[MapRelation],
        relations_master:  List[MapRelation], 
        model: SentenceTransformer,
        metric_configs: List[MetricConfiguration]):
    # metric id 6
    reference_config = next((config for config in metric_configs if config.id == 6), None)
    
    if reference_config:

    
        student_edge_labels = list(set([item.relation_label for item in relations_student]))
        master_edge_labels = list(set([item.relation_label for item in relations_master]))

        similar_labels = []
        for m_label in master_edge_labels:
            similar = list(filter(lambda s_label: words_are_similar(s_label, m_label, model), student_edge_labels))
            if similar:
                similar_labels.append(m_label)
        
        unused = list(set(master_edge_labels) - set(similar_labels))
        score =  1 - (len(unused) / len(master_edge_labels)) if len(unused) > 0 else 1

        return UnusedLabelsMetric(
            label=reference_config.label,
            label_key=reference_config.label_key,
            explanation=reference_config.explanation,
            score=round(score,2),
            unused_labels=unused
        )
