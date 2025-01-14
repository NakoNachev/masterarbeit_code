from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional


@dataclass
class MetricConfiguration:
    id: int
    label: str
    label_key: str
    explanation: str

@dataclass
class Metric:
    label: str = ''
    label_key: str = ''
    explanation: str = ''
    score: int | float = 0

@dataclass
class MissingConnectionsMetric(Metric):
    missing_connections: List[str] = field(default_factory=list)
@dataclass
class MislabeledConnectionsMetric(Metric):
    mislabeled_connections: List[str] = field(default_factory=list)
@dataclass
class ExtraConnectionsMetric(Metric):
    extra_connections: List[str] = field(default_factory=list)

@dataclass
class MissingConceptsMetric(Metric):
    concepts_importance: Dict[str, str] = field(default_factory=dict)
    missing_concepts_importance: Dict[str, str] = field(default_factory=dict)
    missing_concepts: List[str] = field(default_factory=list)

@dataclass
class ExtraConceptsMetric(Metric):
    extra_concepts: List[str] = field(default_factory=list)

@dataclass
class UnusedLabelsMetric(Metric):
    unused_labels: List[str] = field(default_factory=list)

@dataclass
class IsolatedConceptsMetric(Metric):
    isolated_concepts: List[str] = field(default_factory=list)

@dataclass
class EvaluationResult:
    metrics: List[Metric] = field(default_factory=list)

@dataclass
class CentralityMetric(Metric):
    centrality_abs_difference: Dict[str, float] = field(default_factory=dict)

@dataclass
class DensityMetric(Metric):
    student_density: float = 0
    master_density: float = 0

@dataclass
class GraphCyclesMetric(Metric):
    cycles_student: List[Any] = field(default_factory=list)
    cycles_master: List[Any] = field(default_factory=list)