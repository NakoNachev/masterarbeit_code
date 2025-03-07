
from cm_evaluator.evaluators.score_evaluator import calculate_group_score
from cm_evaluator.models.metrics import CentralityMetric, DensityMetric, ExtraConceptsMetric, ExtraConnectionsMetric, GraphCyclesMetric, IsolatedConceptsMetric, Metric, MislabeledConnectionsMetric, MissingConceptsMetric, MissingConnectionsMetric, UnusedLabelsMetric
from cm_evaluator.models.output_model import GroupScores


metrics =  {
    "missing_connections": MissingConnectionsMetric(
        label="Missing Connections",
        label_key="missing_connections",
        explanation="This metric identifies connections in the student's solution that are present in the reference solution but missing in the student's solution.",
        score=0.19,
        missing_connections=[
            "Computer Vision detects Objects",
            "AI applies Computer Vision",
            "Machine Learning utilizes Neural Networks",
            "Natural Language Processing enables Sentiment Analysis",
            "Computer Vision applies Facial Recognition",
            "Machine Learning employs Algorithms",
            "Natural Language Processing supports Language Translation",
            "Computer Vision analyzes Images",
            "Automation integrates Internet of Things (IoT)",
            "Natural Language Processing enables Speech Recognition",
            "Automation includes Robotics",
            "Machine Learning achieves Predictive Analytics",
            "Automation improves Efficiency",
        ],
    ),
    "mislabeled_connections": MislabeledConnectionsMetric(
        label="Mislabeled Connections",
        label_key="mislabeled_connections",
        explanation="This metric identifies connections in the student's solution where the labeling does not match the reference solution.",
        score=1.0,
        mislabeled_connections=[],
    ),
    "extra_connections": ExtraConnectionsMetric(
        label="Extra Connections",
        label_key="extra_connections",
        explanation="This metric shows connections in the student's solution that are not present in the reference solution.",
        score=1.0,
        extra_connections=[],
    ),
    "missing_concepts": MissingConceptsMetric(
        label="Missing Concepts",
        label_key="missing_concepts",
        explanation="This metric identifies concepts that are present in the reference solution but missing in the student's solution.",
        score=0.24,
        concepts_importance={
            "AI": "medium",
            "Automation": "high",
            "Machine Learning": "high",
            "Natural Language Processing": "high",
            "Computer Vision": "high",
            "Robotics": "low",
            "Internet of Things (IoT)": "low",
            "Efficiency": "low",
            "Neural Networks": "low",
            "Algorithms": "low",
            "Predictive Analytics": "low",
            "Speech Recognition": "low",
            "Sentiment Analysis": "low",
            "Language Translation": "low",
            "Objects": "low",
            "Images": "low",
            "Facial Recognition": "low",
        },
        missing_concepts_importance={
            "Computer Vision": "high",
            "Robotics": "low",
            "Internet of Things (IoT)": "low",
            "Efficiency": "low",
            "Neural Networks": "low",
            "Algorithms": "low",
            "Predictive Analytics": "low",
            "Speech Recognition": "low",
            "Sentiment Analysis": "low",
            "Language Translation": "low",
            "Objects": "low",
            "Images": "low",
            "Facial Recognition": "low",
        },
        missing_concepts=[
            "Computer Vision",
            "Speech Recognition",
            "Neural Networks",
            "Sentiment Analysis",
            "Images",
            "Internet of Things (IoT)",
            "Objects",
            "Language Translation",
            "Predictive Analytics",
            "Facial Recognition",
            "Efficiency",
            "Robotics",
            "Algorithms",
        ],
    ),
    "extra_concepts": ExtraConceptsMetric(
        label="Extra Concepts",
        label_key="extra_concepts",
        explanation="This metric shows concepts in the student's solution that are not part of the reference solution.",
        score=1,
        extra_concepts=[],
    ),
    "unused_labels": UnusedLabelsMetric(
        label="Unused Labels",
        label_key="unused_labels",
        explanation="This metric shows labels that are present in the reference solution but not used in the student's solution.",
        score=0.5,
        unused_labels=[
            "achieves",
            "analyzes",
            "detects",
            "integrates",
            "supports",
            "improves",
        ],
    ),
    "isolated_concepts": IsolatedConceptsMetric(
        label="Isolated Concepts",
        label_key="isolated_concepts",
        explanation="This metric identifies concepts in the student's solution that have no connections to other concepts, compared to the reference solution.",
        score=1,
        isolated_concepts=[],
    ),
    "topologic_similarity": Metric(
        label="Topological Similarity",
        label_key="topologic_similarity",
        explanation="This metric compares the structure of the entire networks using the Jaccard coefficient.",
        score=0.2,
    ),
    "centrality_similarity": CentralityMetric(
        label="Centrality Similarity",
        label_key="centrality_similarity",
        explanation="This metric calculates the centrality (importance) of concepts in both maps and compares them.",
        score=0.75,
        centrality_abs_difference={
            "AI": 0.75,
            "Automation": 0.0833,
            "Machine Learning": 0.0833,
            "Natural Language Processing": 0.0833,
        },
    ),
    "connection_density": DensityMetric(
        label="Connection Density",
        label_key="connection_density",
        explanation="The connection density of a network measures how strongly the nodes (in this case, the student's concepts) are connected to each other. It indicates how many actual connections exist compared to the maximum possible connections. A higher density means the concepts are more interconnected.",
        score=0.5,
        student_density=0.25,
        master_density=0.0588,
    ),
    "graph_cycles": GraphCyclesMetric(
        label="Graph Cycles",
        label_key="graph_cycles",
        explanation="A cycle in a graph occurs when a sequence of edges and nodes forms a closed path, meaning the starting node of the sequence is also the end node. In a concept map, cycles should generally be avoided as they can complicate the hierarchical structure and understanding of the relationships between concepts. An acyclic structure promotes clear and orderly comprehension of the concepts.",
        score=1,
        cycles_student=[],
        cycles_master=[],
    ),
}

def test_calculate_group_score():
   result = calculate_group_score(metrics)

   expected = GroupScores(
    concept_inclusion_score= 0.47,
    connection_accuracy_score= 0.49,
    structural_integrity_score= 0.76,
    total_score= 0.51
   )

   assert result == expected