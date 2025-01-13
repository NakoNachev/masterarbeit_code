from dataclasses import dataclass, field
from typing import Dict, List

from cm_evaluator.models.metrics import Metric


@dataclass
class MapInfo:
    relations_as_sentences: List[str] = field(default_factory=list)

@dataclass
class Maps:
    student: MapInfo = field(default_factory=MapInfo)
    reference: MapInfo = field(default_factory=MapInfo)

@dataclass
class GroupScores:
    concept_inclusion_score: float
    connection_accuracy_score: float
    structural_integrity_score: float
    total_score: float

@dataclass
class EvaluationOutput:
    metrics: Dict[str, Metric]
    maps: Maps
    group_scores: GroupScores

