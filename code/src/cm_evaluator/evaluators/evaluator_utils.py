
from typing import Dict, List

from dataclasses import asdict

from cm_evaluator.evaluators.score_evaluator import calculate_group_score
from cm_evaluator.models.metrics import Metric, MetricConfiguration
from cm_evaluator.models.helper_models import MapRelation
from cm_evaluator.models.output_model import EvaluationOutput, MapInfo, Maps
from cm_evaluator.utilities import get_sentence_representations
from cm_evaluator.logging.base_logger import logger

def create_output(student_relations: List[MapRelation], 
                  master_relations: List[MapRelation], 
                  output_metrics: List[Metric]):
    maps = Maps(
        student=MapInfo(relations_as_sentences=get_sentence_representations(student_relations)),
        reference=MapInfo(relations_as_sentences=get_sentence_representations(master_relations)),
    )
    metrics: Dict[str, Metric] = {}
    for _, val in enumerate(output_metrics):
        metrics[val.label_key] = val

    # needs as dict since class instances are not serializable
    return asdict(EvaluationOutput(
        metrics=metrics,
        maps=maps,
        group_scores=calculate_group_score(metrics)
    ))