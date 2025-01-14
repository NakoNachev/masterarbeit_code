from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Dict, List

from cm_evaluator.decorators.decorators import time_and_log_decorator
from cm_evaluator.models.helper_models import WeightsConfig
from cm_evaluator.models.output_model import GroupScores

from cm_evaluator.models.metrics import Metric

@time_and_log_decorator
def calculate_group_score(output: Dict[str, Metric]) -> GroupScores:
    with open(files("cm_evaluator.data").joinpath("metrics_weights.json"), 'r', encoding='utf-8') as fd: # type: ignore
        weights_config: List[WeightsConfig] = [WeightsConfig(**data) for data in json.load(fd)]
    group_scores = {}
    # calculate scores for each group
    for config in weights_config:
        group_score = 0
        for metric_label, weight in config.element_weights.items():
            group_score += output[metric_label].score * weight
        group_scores[f'group{config.group_id}_score'] = round(group_score, 2)

    # calculate total score (as a weighted sum of each group)
    total_score = sum(
        group_scores[f'group{config.group_id}_score'] * config.weight
        for config in weights_config
    )

    return GroupScores(
        concept_inclusion_score=group_scores.get('group1_score', 0),
        connection_accuracy_score=group_scores.get('group2_score', 0),
        structural_integrity_score=group_scores.get('group3_score', 0),
        total_score=round(total_score, 2)
    )
