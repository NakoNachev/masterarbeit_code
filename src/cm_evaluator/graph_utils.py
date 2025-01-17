from typing import List

import numpy as np
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer

from cm_evaluator.models.helper_models import MapRelation
from cm_evaluator.utilities import common_words, get_intersection_of_similar_sentences, get_union_of_similar_sentences


def create_graph_from_data(data: List[MapRelation], nodes: List[str]):
    G = nx.DiGraph()

    nodes_with_relations = set()

    for mapped_relation in data:
        nodes_with_relations.add(mapped_relation.conceptA)
        nodes_with_relations.add(mapped_relation.conceptB)
        G.add_edge(
            mapped_relation.conceptA, 
            mapped_relation.conceptB, 
            label=mapped_relation.relation_label, 
            arrows="to")
    

    isolated_nodes = set(nodes) - nodes_with_relations
    for node in isolated_nodes:
        G.add_node(node)
    return G

def find_graph_cyles(graph: nx.Graph) -> None | List:
    """ find if there are any cycles in the graph
    a cycle is for example B -> C and C -> B"""
    try:
        cycles = list(nx.simple_cycles(graph))
        return cycles
    except nx.exception.NetworkXNoCycle:
        return None
    
def find_isolates(graph: nx.Graph):
    return list(nx.isolates(graph))

def get_adjacency_matrix_dense(graph: nx.Graph) -> np.ndarray:
    return nx.adjacency_matrix(graph).todense()

def get_adjacency_matrix_pd(graph: nx.Graph) -> pd.DataFrame:
    matrix = get_adjacency_matrix_dense(graph)
    nodes = list(graph.nodes)
    return pd.DataFrame(matrix, index=nodes, columns=nodes)

def get_predecessors(graph: nx.DiGraph, node: str):
    """ returns the nodes pointing towards the node (incomming) """
    if not graph.has_node(node):
        raise Exception(f'Provided node {node} not found in graph')
    if not isinstance(graph, nx.DiGraph):
        raise Exception('Graph object should be a directional graph')
    return list(graph.predecessors(node))

def get_successors(graph: nx.DiGraph, node: str):
    """ returns the nodes to which the node points (outcomming) """
    if not graph.has_node(node):
        raise Exception(f'Provided node {node} not found in graph')
    if not isinstance(graph, nx.DiGraph):
        raise Exception('Graph object should be a directional graph')
    return list(graph.successors(node))

def get_incoming_outcoming_and_labels(graph: nx.DiGraph, node: str):
    incoming = get_predecessors(graph, node)
    outcoming = get_successors(graph, node)
    node_info = {"incoming": [], "outcoming": []}
    for n in incoming:
        label = graph.get_edge_data(n, node)['label']
        node_info["incoming"].append({"label": label, "origin": n})
    for n in outcoming:
        label = graph.get_edge_data(node, n)['label']
        node_info["outcoming"].append({"label": label, "target": n})

    return node_info

def get_incoming_outcoming_labels_whole_graph(graph: nx.DiGraph):
    nodes = list(graph.nodes)
    info = {}
    for node in nodes:
        info[node] = get_incoming_outcoming_and_labels(graph, node)
    return info

def adjacency_matrix_similarity(graph1, graph2):
    """ computes the topologic similarity between two graphs based on jaccard index"""
    # Compute adjacency matrices
    adj_matrix1 = nx.adjacency_matrix(graph1).todense()
    adj_matrix2 = nx.adjacency_matrix(graph2).todense()

    intersection = np.logical_and(adj_matrix1, adj_matrix2).sum()
    union = np.logical_or(adj_matrix1, adj_matrix2).sum()
    return intersection / union

 
def jaccard_similarity_with_synonyms(student_relations: List[MapRelation], master_relations: List[MapRelation],
                                    model: SentenceTransformer):

    # Compute intersection by checking for equivalent edges
    intersection_count = get_intersection_of_similar_sentences(student_relations, master_relations, model)
    union = get_union_of_similar_sentences(student_relations, master_relations, model)
    
    # Avoid division by zero
    if not union:
        return 0.0

    jaccard_index = intersection_count / union
    return round(jaccard_index,2)

def centrality_similarity(student_graph: nx.DiGraph, reference_graph: nx.DiGraph, model: SentenceTransformer):
    student_centrality = nx.degree_centrality(student_graph)
    reference_centrality = nx.degree_centrality(reference_graph)

    common_nodes = common_words(student_graph.nodes(), reference_graph.nodes(), model)
    abs_differences = {node: abs(student_centrality.get(node, 0) - reference_centrality.get(node, 0)) 
                       for node in common_nodes}
    if reference_centrality and student_centrality:

        similarity = sum(1 - abs_differences.get(node, 0)
                        for node in common_nodes) / len(common_nodes)
        return similarity
    else:
        return 0



def density_difference(g1: nx.DiGraph, g2: nx.DiGraph):
    density_difference = nx.density(g1) > nx.density(g2) 
    match density_difference:
        case True:
            return 1
        case False:
            if nx.density(g1) == nx.density(g2):
                return 0.5
            else:
                return 0
