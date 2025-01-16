# Concepte und Verbindungen sind vorgegeben, die Studierenden sollen diese logisch miteinander verbinden.
import importlib
from pathlib import Path
from typing import List, Literal
import json
from importlib.resources import files

from cm_evaluator.decorators.decorators import time_and_log_decorator
from cm_evaluator.evaluators.common import common_eval_functions
from cm_evaluator.models.helper_models import MapRelation
from cm_evaluator.models.user_data import MapOutput
import networkx as nx
from sentence_transformers import SentenceTransformer

from cm_evaluator.evaluators.evaluator_utils import create_output
from cm_evaluator.models.metrics import Metric, MetricConfiguration, MislabeledConnectionsMetric, MissingConnectionsMetric, UnusedLabelsMetric
from cm_evaluator.graph_utils import create_graph_from_data
from cm_evaluator.logging.base_logger import logger

def evaluate(
        cm_student: MapOutput, 
        cm_master: MapOutput, 
        model: SentenceTransformer,
        language: Literal["de", "en"] = "en",
        dump_output: bool = False):
    student_concepts = cm_student.get_node_names()
    master_concepts = cm_master.get_node_names()
    student_relations = cm_student.convert_to_map_relation()
    master_relations = cm_master.convert_to_map_relation()
    G_student = create_graph_from_data(student_relations, student_concepts)
    G_master = create_graph_from_data(master_relations, master_concepts)
    metric_configs = []
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_json = output_dir / f'output_{cm_student.user}.json'
    with open(files("cm_evaluator.data").joinpath(f'metrics_{language}.json'), 'r', encoding='utf-8') as fd: # type: ignore
        metric_configs: List[MetricConfiguration] = [MetricConfiguration(**data) for data in json.load(fd)]

    logger.debug("Read up the metrics configuration file")
    metric_function_mapper = {
        "1": {"func_name": "missing_connections", "args": [student_relations, master_relations, metric_configs]},
        "2": {"func_name": "mislabeled_connections", "args": [student_relations, master_relations, metric_configs]},
        "3": {"func_name": "extra_connections", "args": [G_student, G_master, metric_configs], "common_func": True},
        "4": {"func_name": "missing_concepts", "args": [student_concepts, master_concepts, G_master, language, metric_configs], "common_func": True},
        "5": {"func_name": "extra_concepts", "args": [student_concepts, master_concepts, metric_configs], "common_func": True},
        "6": {"func_name": "unused_edge_labels", "args": [student_relations, master_relations, metric_configs]},
        "7": {"func_name": "isolates", "args": [G_student, metric_configs], "common_func": True},
        "8": {"func_name": "topologic_similarity", "args": [student_relations, master_relations, G_student, G_master, metric_configs, model], "common_func": True},
        "9": {"func_name": "get_centrality_similarity", "args": [G_student, G_master, metric_configs, model], "common_func": True},
        "10": {"func_name": "get_density_difference", "args": [G_student, G_master, metric_configs], "common_func": True},
        "11": {"func_name": "get_cycles", "args": [G_student, G_master, metric_configs], "common_func": True},
    }

    # collect the evaluated metrics
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
            json.dump(output, fd)
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
def missing_connections(
        student_relations: List[MapRelation], 
        master_relations: List[MapRelation], 
        metric_configs: List[MetricConfiguration]):
    # metric id 1
    reference_config = next((config for config in metric_configs 
                             if config.id == 1), None)
    missing = []
    if reference_config:

        for relation in master_relations:
            if relation not in student_relations:
                missing.append(relation.get_sentence_repr())

        score = 1 - (len(missing) / len(master_relations))

        metric = MissingConnectionsMetric(
            label=reference_config.label,
            label_key="missing_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            missing_connections=missing
        )
        return metric

@time_and_log_decorator
def mislabeled_connections(
        student_relations: List[MapRelation], 
        master_relations: List[MapRelation],
        metric_configs: List[MetricConfiguration]):
    ''' Finds all connections between two nodes where the connection label does not match '''
    # metric id 2
    reference_config = next((config for config in metric_configs if config.id == 2), None)
    if reference_config:

        if len(student_relations) == 0:
            return MislabeledConnectionsMetric(
            label=reference_config.label,
            label_key="mislabeled_connections",
            explanation=reference_config.explanation,
            score=0,
            mislabeled_connections=[]
            )
        mislabeled_connections = []
        for st_rel in student_relations:
            for m_rel in master_relations:
                if st_rel.conceptA == m_rel.conceptA and st_rel.conceptB == m_rel.conceptB and st_rel.relation_label != m_rel.relation_label:
                    mislabeled_connections.append(f'{st_rel.get_sentence_repr()} ({m_rel.relation_label})')

        score = 1 - (len(mislabeled_connections) / len(master_relations))

        return MislabeledConnectionsMetric(
            label=reference_config.label,
            label_key="mislabeled_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            mislabeled_connections=mislabeled_connections
        )
    
@time_and_log_decorator  
def unused_edge_labels(
        relations_student: List[MapRelation], 
        relations_master: List[MapRelation], 
        metric_configs: List[MetricConfiguration]):
    # metric id 6
    reference_config = next((config for config in metric_configs if config.id == 6), None)
    if reference_config:

        master_labels = [item.relation_label for item in relations_master]
        student_labels = [item.relation_label for item in relations_student]
        unused = set(master_labels) - set(student_labels)

        score = 1 - (len(unused) / len(master_labels))

        return UnusedLabelsMetric(
            label=reference_config.label,
            label_key=reference_config.label_key,
            explanation=reference_config.explanation,
            score=round(score,2),
            unused_labels=list(unused)
        )
