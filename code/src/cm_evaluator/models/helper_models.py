from dataclasses import dataclass
from typing import Dict


class MapRelation:
    conceptA: str
    conceptB: str
    relation_label: str

    def __init__(self, conceptA, relation_label, conceptB):
        self.conceptA = conceptA
        self.relation_label = relation_label
        self.conceptB = conceptB

    def get_sentence_repr(self) -> str:
        return " ".join([self.conceptA, self.relation_label, self.conceptB])
    
    def __eq__(self, other):
        return (self.conceptA == other.conceptA 
                and self.conceptB == other.conceptB 
                and self.relation_label == other.relation_label)

@dataclass
class WeightsConfig:
    group_id: int
    weight: float
    element_weights: Dict[str, float]