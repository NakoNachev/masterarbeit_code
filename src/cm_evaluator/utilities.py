from typing import Dict, List, Literal
import csv

import numpy as np
import pandas as pd
import networkx as nx
from numpy.linalg import norm
from spacy.language import Language
from scipy.stats import pearsonr
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity

from cm_evaluator.models.helper_models import MapRelation

def eucledian_distance(v1: List[int], v2: List[int]):
    if len(v1) != len(v2):
        raise ValueError('Cannot calculate eucledian distance of vectors with different length')
    distance = np.array(v1) - np.array(v2)
    return norm(distance)

def get_union_vector(s1: List[str], s2: List[str]):
    union = []
    for item in s1: 
        if item not in union:
            union.append(item)

    for item in s2: 
        if item not in union:
            union.append(item)
    return union

def get_order_vector(union_vec: List[str], s: List[str]):
    order_vector = []
    for item in union_vec:
        if item in s:
            order_vector.append(s.index(item) + 1)
        else:
            order_vector.append(0)
    return order_vector


def get_cosine_sim(v1: List[int], v2: List[int]):
    a = np.array(v1)
    b = np.array(v2)
    return round(np.dot(a,b) / (norm(a) * norm(b)), 2)


def get_pearson_corr(v1: List[int], v2: List[int]):
    """ could later be used for evaluation of an annotated dataset vs result from system"""
    return pearsonr(v1, v2)
    
def get_cosine_sim_spacy(nlp: Language, s1: str, s2: str):
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    return doc1.similarity(doc2)

def fetch_input_examples(file_path) -> List[InputExample]:
    examples = []
    with open(file_path, 'r', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=";")
        next(reader)
        for line in reader:
            if pd.notna(line[0]) and pd.notna(line[1]) and pd.notna(line[2]):
                score = line[2]
                examples.append(InputExample(texts=[line[0], line[1]], label=float(score)))
    return examples

def extract_pos(nlp: Language, sentence: str, pos: str):
    doc = nlp(sentence)
    return [token.pos_ for token in doc if token.pos_ == pos]

def get_nearest_sentence_matrix(sentences1: List[str], sentences2: List[str], model: SentenceTransformer):
    embeddings_s1 = np.array([model.encode(sentence) for sentence in sentences1])
    embeddings_s2 = np.array([model.encode(sentence) for sentence in sentences2])

    similarities = cosine_similarity(embeddings_s1.reshape(len(sentences1), -1), embeddings_s2.reshape(len(sentences2), -1))
    return similarities

def extract_misinterpreted_sentences(
        sentences1: List[str], 
        sentences2: List[str], 
        model: SentenceTransformer, 
        threshold = 0.8):
    """ finds all sentences from the student which are not similar to either one
    of the master solution. Returns a dict with the sentence and the nearest sentence
    from the master solution with its similarity"""
    misinterpreted = {}
    matrix = get_nearest_sentence_matrix(sentences1, sentences2, model)
    for i, sentence in enumerate(sentences1):
        best_match_i = np.argmax(matrix[i])
        best_match_sentence = sentences2[best_match_i]
        similarity = round(matrix[i][best_match_i],2)

        if similarity != 1.0 and similarity < threshold:
            misinterpreted[sentence] = (best_match_sentence, similarity)

    return misinterpreted

def find_missing_concepts(concepts_master: List[str], concepts_student: List[str]) -> set:
    return set(concepts_master) - set(concepts_student)

def extract_concepts(data: List[MapRelation]):
    concepts = []
    for item in data:
        concepts.append(item.conceptA)
        concepts.append(item.conceptB)
    return list(set(concepts))

def words_are_similar(w1: str, w2: str, model: SentenceTransformer,
                      is_label: bool = False, threshold=0.8):
    template_sentence="Das Wort ist "
    if (is_label):
        template_sentence = "Konzept A {label} Konzept B"
        s1 = template_sentence.format(label=w1)
        s2 = template_sentence.format(label=w2)
    else:
        s1 = f'{template_sentence}{w1}'
        s2 = f'{template_sentence}{w2}'
    
    v1 = model.encode(s1)
    v2 = model.encode(s2)
    return util.cos_sim(v1, v2) >= threshold

def get_most_similar_word(
        word: str, 
        candidates: List[str], 
        model:SentenceTransformer) -> str | None:

    if word in candidates:
        return word
    
    most_similar = list(filter(
        lambda x: words_are_similar(word, x, model),
        candidates))
    
    if len(most_similar) == 1:
        return most_similar[0]
    if len(most_similar) == 0:
        return None
    
    return None


def common_words(words1: List[str], words2: List[str], model: SentenceTransformer, is_label: bool = False) -> List[str]:
    common = set(words1) & set(words2)
    if not model:
        return list(common)
    similar_concepts = []
    for word in list(words1):
        similar = list(filter(lambda w_student: words_are_similar(word, w_student, model, is_label), words2))
        if similar:
            similar_concepts.append(word)
    return similar_concepts

# def get_remaining_master_sentences(
#         master_sentences: List[str], 
#         student_sentences: List[str], 
#         model: SentenceTransformer, 
#         threshold: float = 0.8):
#     """ Returns the sentences from the master solution that were not correctly identified
#     Correctly identified sentences = sentence which has either 1.0 or >= threshold similarity
#     with at least one of the master solution sentences"""
#     correct_sentences = []

#     if len(student_sentences) == 0:
#         return set(master_sentences)

#     matrix = get_nearest_sentence_matrix(master_sentences, student_sentences, model)
#     print("matrix", matrix)
    
#     for i, sentence in enumerate(master_sentences):
#         best_match_i = np.argmax(matrix[i])
#         similarity = round(matrix[i][best_match_i],2)
#         if similarity == 1 or similarity >= threshold:
#             correct_sentences.append(sentence)
#     return set(master_sentences) - set(correct_sentences)


def get_remaining_master_sentences(
        master_sentences: List[str], 
        student_sentences: List[str], 
        model: SentenceTransformer, 
        threshold: float = 0.8):
    """
    Returns the master sentences that do not have a sufficiently similar match in the student solution.
    Ensures that each student sentence can only match one master sentence.
    """
    if len(student_sentences) == 0:
        return set(master_sentences)

    matrix = get_nearest_sentence_matrix(master_sentences, student_sentences, model)
    correct_sentences = set()
    used_column_indexes = set()

    for i, sentence in enumerate(master_sentences):
        best_match_idx = np.argmax(matrix[i])
        similarity = round(matrix[i][best_match_idx],2)

        if similarity >= threshold and best_match_idx not in used_column_indexes:
            correct_sentences.add(sentence)
            used_column_indexes.add(best_match_idx)

    return set(master_sentences) - correct_sentences


def assign_node_importance_weights(graph: nx.DiGraph) -> Dict[str, float]:
    ''' Calculates the node importance weights based on the ingoing and outgoing nodes'''
    # Default weights for in-degree and out-degree
    weights_config = {
        'in-degree': 0.7,
        'out-degree': 0.3
    }
    
    # Calculate weighted degree for each node
    weighted_degrees = {}
    for node in graph.nodes:
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        weighted_degrees[node] = (
            weights_config['in-degree'] * in_deg +
            weights_config['out-degree'] * out_deg
        )
    
    # Sum of all weighted degrees (for normalization)
    total_weighted_degree = sum(weighted_degrees.values())
    
    # Normalize scores to ensure the sum is 1
    if total_weighted_degree > 0:
        normalized_scores = {node: round((score / total_weighted_degree),2) for node, score in weighted_degrees.items()}
    else:
        # If no edges exist, assign equal scores to all nodes
        normalized_scores = {node: 1 / len(graph.nodes) for node in graph.nodes}
    
    return normalized_scores

def get_nodes_importance(graph: nx.DiGraph, language: Literal["de", "en"]) -> Dict[str, str]:

    nodes_importance_weights = assign_node_importance_weights(graph)

    # calculate percentiles to rang weights
    all_scores = np.array(list(set(nodes_importance_weights.values())))
    low_threshold = np.percentile(all_scores, 33)  
    medium_threshold = np.percentile(all_scores, 67)

    def categorize_priority_dynamic(score: float):
        if score >= medium_threshold:
            return "high" if language == "en" else "hoch"
        elif score >= low_threshold:
            return "medium" if language == "en" else "mittel"
        else:
            return "low" if language == "en" else "niedrig"
        
    nodes_importance = { 
        concept: categorize_priority_dynamic(importance)
        for concept, importance in nodes_importance_weights.items()
    }

    return nodes_importance


def get_sentence_representations(relations: List[MapRelation]):
    return [relation.get_sentence_repr() for relation in relations]


def get_intersection_of_similar_sentences(student_relations: List[MapRelation], 
        master_relations: List[MapRelation], model: SentenceTransformer, threshold: float = 0.8):
    student_sentences = [item.get_sentence_repr() for item in student_relations]
    master_sentences = [item.get_sentence_repr() for item in master_relations]

    student_encodes = {s: model.encode(s) for s in student_sentences}
    master_encodes = {s: model.encode(s) for s in master_sentences}

    intersection_len = 0
    matched_master_sentences = set()
    for student_sentence in student_sentences:
        s1_encode = student_encodes.get(student_sentence, None)
        for master_sentence in master_sentences:
            if master_sentence in matched_master_sentences:
                continue 
            s2_encode = master_encodes.get(master_sentence, None)
            if util.cos_sim(s1_encode, s2_encode) >= threshold:
                intersection_len += 1
                matched_master_sentences.add(master_sentence)
                break

    return intersection_len

def get_union_of_similar_sentences(student_relations: List[MapRelation], 
        master_relations: List[MapRelation], model: SentenceTransformer, threshold: float = 0.8):
    student_sentences = [item.get_sentence_repr() for item in student_relations]
    master_sentences = [item.get_sentence_repr() for item in master_relations]

    student_encodes = {s: model.encode(s) for s in student_sentences}
    master_encodes = {s: model.encode(s) for s in master_sentences}

    union_sentences = set(student_sentences)

    # Add master sentences if they are not semantically similar to any in the union
    for master_sentence in master_sentences:
        m_encode = master_encodes.get(master_sentence, None)
        is_identical_to_any = False
        for student_sentence in student_sentences:
            s_encode = student_encodes.get(student_sentence, None)
            if util.cos_sim(m_encode, s_encode) >= threshold:
                is_identical_to_any = True
        if not is_identical_to_any:
            union_sentences.add(master_sentence)


    return len(union_sentences)