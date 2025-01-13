from typing import Dict, List, Literal

from cm_evaluator.models.helper_models import MapRelation
import networkx as nx
from sentence_transformers import SentenceTransformer

from cm_evaluator.graph_utils import find_graph_cyles, find_isolates, jaccard_similarity_with_synonyms
from cm_evaluator.models.metrics import CentralityMetric, DensityMetric, ExtraConceptsMetric, ExtraConnectionsMetric, GraphCyclesMetric, IsolatedConceptsMetric, Metric, MetricConfiguration, MissingConceptsMetric, MissingConnectionsMetric
from cm_evaluator.utilities import common_words, find_missing_concepts, get_nodes_importance, get_remaining_master_sentences, assign_node_importance_weights
from cm_evaluator.decorators.decorators import time_and_log_decorator

@time_and_log_decorator
def missing_concepts(
        concepts_student: List[str], 
        concepts_master: List[str],
        graph_master: nx.DiGraph,
        language: Literal["en", "de"],
        metric_configs: List[MetricConfiguration]):
    # metric id 4
    # concepts are alrady given, therefore a straightforward check is necessary
    missing = list(find_missing_concepts(concepts_master, concepts_student))

    # save importance of nodes
    nodes_importance = get_nodes_importance(graph_master, language)
    reference_config = next((config for config in metric_configs if config.id == 4), None)

    if reference_config:
        score = (1 - (len(missing) / len(concepts_master))) if len(concepts_master) > 0 else 0

        return MissingConceptsMetric(
            label=reference_config.label,
            label_key="missing_concepts",
            explanation=reference_config.explanation,
            score=round(score,2),
            missing_concepts=missing,
            concepts_importance=nodes_importance,
            missing_concepts_importance={key: value for key, value in nodes_importance.items() if key in missing})

@time_and_log_decorator
def extra_concepts(
        student_concepts: List[str], 
        master_concepts: List[str],
        metric_configs: List[MetricConfiguration]):
    # metric id 5 
    extra_concepts = list(set(student_concepts) - set(master_concepts))
    score =  1 - (len(extra_concepts) / len(student_concepts)) if len(extra_concepts) > 0 else 1
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


        return ExtraConceptsMetric(
            label=reference_config.label,
            label_key="extra_concepts",
            explanation=reference_config.explanation,
            score=round(score,2),
            extra_concepts=extra_concepts
        )

@time_and_log_decorator
def extra_connections(
        student_map: nx.DiGraph, 
        master_map: nx.DiGraph, 
        metric_configs: List[MetricConfiguration]):
    # metric id 3
    extra_edges = []

    reference_config = next((config for config in metric_configs if config.id == 3), None)
    if reference_config:

        if len(student_map.edges()) == 0:

            return ExtraConnectionsMetric(
                label=reference_config.label,
                label_key="extra_connections",
                explanation=reference_config.explanation,
                score=0,
                extra_connections=[f"{edge[0]} {student_map.get_edge_data(edge[0], edge[1])['label']} {edge[1]}" for edge in extra_edges]
            )

        for edge in student_map.edges():
            u, v = edge
            if not (master_map.has_edge(u, v) or master_map.has_edge(v, u)):
                extra_edges.append(edge)

        score = 1 - (len(extra_edges) / len(student_map.edges()))

        return ExtraConnectionsMetric(
            label=reference_config.label,
            label_key="extra_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            extra_connections=[f"{edge[0]} {student_map.get_edge_data(edge[0], edge[1])['label']} {edge[1]}" for edge in extra_edges]
        )

@time_and_log_decorator
def isolates(
        graph: nx.DiGraph, 
        metric_configs: List[MetricConfiguration]):
    # metric id 7
    reference_config = next((config for config in metric_configs if config.id == 7), None)
    if reference_config:

        if len(list(graph.nodes)) == 0:
            return IsolatedConceptsMetric(
                label=reference_config.label,
                label_key="isolated_concepts",
                explanation=reference_config.explanation,
                score=0,
                isolated_concepts=[]
            )

        isolates = find_isolates(graph)
        concepts_student = list(graph.nodes)
        score = 1 - (len(isolates) / len(concepts_student)) if len(isolates) != 0 else 1
        return IsolatedConceptsMetric(
            label=reference_config.label,
            label_key="isolated_concepts",
            explanation=reference_config.explanation,
            score=round(score,2),
            isolated_concepts=list(isolates)
        )

@time_and_log_decorator
def topologic_similarity(student_relations: List[MapRelation], master_relations: List[MapRelation],
                         graph_student: nx.DiGraph, graph_master: nx.DiGraph,
                         metric_configs: List[MetricConfiguration], model: SentenceTransformer):
    # metric id 8
    reference_config = next((config for config in metric_configs if config.id == 8 ), None)
    if reference_config:
        if len(list(graph_student.nodes())) == 0 or len(list(graph_student.edges())) == 0:
            return Metric(
                label=reference_config.label,
                label_key="topologic_similarity",
                explanation=reference_config.explanation,
                score=0,
            )
        score = jaccard_similarity_with_synonyms(student_relations, master_relations, model)
        return Metric(
            label=reference_config.label,
            label_key="topologic_similarity",
            explanation=reference_config.explanation,
            score=score,
        )

@time_and_log_decorator
def get_centrality_similarity(graph_student: nx.DiGraph, graph_master: nx.DiGraph, metric_configs: List[MetricConfiguration], model: SentenceTransformer):
    # metric id 9
    reference_config = next((config for config in metric_configs if config.id == 9 ), None)
    if reference_config:

        student_centrality = nx.degree_centrality(graph_student)
        reference_centrality = nx.degree_centrality(graph_master)

        common_nodes = common_words(graph_student.nodes(), graph_master.nodes(), model)
        abs_differences: Dict[str, float] = {node: round(abs(student_centrality.get(node, 0) - reference_centrality.get(node, 0)),4) 
                        for node in common_nodes}
        if reference_centrality and student_centrality and len(common_nodes) != 0:
            similarity = sum(1 - abs_differences.get(node, 0)
                            for node in common_nodes) / len(common_nodes)
            score = similarity
        else:
            score = 0
        
        
        return CentralityMetric(
            label=reference_config.label,
            label_key="centrality_similarity",
            explanation=reference_config.explanation,
            score=round(score,2),
            centrality_abs_difference=abs_differences
        )

@time_and_log_decorator
def get_density_difference(graph_student: nx.DiGraph, graph_master: nx.DiGraph, metric_configs: List[MetricConfiguration]):
    # Metric ID 10
    reference_config = next((config for config in metric_configs if config.id == 10), None)
    if reference_config:

        density_student = nx.density(graph_student)
        density_master = nx.density(graph_master)
        difference = abs(density_student - density_master)
        if len(list(graph_student.nodes())) == 0:
            return DensityMetric(
                label=reference_config.label,
                label_key="connection_density",
                explanation=reference_config.explanation,
                score=0,
                student_density=round(density_student, 4),
                master_density=round(density_master,4)
            )

        def get_score(difference: float):
            ranges = {
                (0.0, 0.05): 1,
                (0.05, 0.1): 0.9,
                (0.1, 0.15): 0.7,
                (0.15, 0.2): 0.5,
                (0.2, 1.0): 0,
            }
            for (low, high), result in ranges.items():
                if low <= difference < high:
                    return result

        #TODO: does it make sense to add some form of a scaling factor
        # graphs with less nodes will be more sensitive to changes in edges
        score = get_score(difference)

        if score is not None:
            return DensityMetric(
                label=reference_config.label,
                label_key="connection_density",
                explanation=reference_config.explanation,
                score=round(score, 2),
                student_density=round(density_student, 4),
                master_density=round(density_master,4)
            )

@time_and_log_decorator
def get_cycles(graph_student: nx.DiGraph, graph_master: nx.DiGraph, metric_configs: List[MetricConfiguration]):
    # Calculate cycles
    reference_config = next((config for config in metric_configs if config.id == 11), None)
    if reference_config:

        if len(graph_student.edges()) == 0 or len(graph_student.nodes()) == 0:
            return GraphCyclesMetric(
                label=reference_config.label,
                label_key="graph_cycles",
                explanation=reference_config.explanation,
                score=0,
                cycles_student = [],
                cycles_master = []
            )

        cycles_student = find_graph_cyles(graph_student)
        cycles_master = find_graph_cyles(graph_master)
        
        abs_difference = abs(len(cycles_student) if cycles_student else 0 - len(cycles_master) if cycles_master else 0)
        scores_schema = {0: 1, 1: 0.75, 2: 0.5, 3: 0.25}
        score = 0
        scoring = scores_schema.get(abs_difference, None)
        if score is not None:
            score = scoring
        return GraphCyclesMetric(
            label=reference_config.label,
            label_key="graph_cycles",
            explanation=reference_config.explanation,
            score=round(score, 2),
            cycles_student=cycles_student if cycles_student else [],
            cycles_master=cycles_master if cycles_master else []
        )

@time_and_log_decorator
def missing_connections(
        student_relations: List[MapRelation], 
        master_relations: List[MapRelation], 
        model: SentenceTransformer,
        metric_configs: List[MetricConfiguration]):
    # metric id 1
    reference_config = next((config for config in metric_configs if config.id == 1), None)
    if reference_config:

        master_sentences = [item.get_sentence_repr() for item in master_relations]
        student_sentences = [item.get_sentence_repr() for item in student_relations]
        missing = list(get_remaining_master_sentences(master_sentences, student_sentences, model))
        score = 1 - (len(missing) / len(master_relations))

        return MissingConnectionsMetric(
            label=reference_config.label,
            label_key="missing_connections",
            explanation=reference_config.explanation,
            score=round(score,2),
            missing_connections=missing
        )