from cm_evaluator.utilities import get_remaining_master_sentences


def test_get_remaining_master_sentences(model):
    reference_sentences =  ["AI enables Automation",
        "AI uses Machine Learning",
        "AI includes Natural Language Processing",
        "AI applies Computer Vision",
        "Automation includes Robotics",
        "Automation integrates Internet of Things (IoT)",
        "Automation improves Efficiency",
        "Machine Learning utilizes Neural Networks",
        "Machine Learning employs Algorithms",
        "Machine Learning achieves Predictive Analytics",
        "Natural Language Processing enables Speech Recognition",
        "Natural Language Processing enables Sentiment Analysis",
        "Natural Language Processing supports Language Translation",
        "Computer Vision detects Objects",
        "Computer Vision analyzes Images",
        "Computer Vision applies Facial Recognition"]
    
    student_sentences = [
        "AI enables Automation",
        "AI uses Machine Learning",
        "AI includes Natural Language Processing"
      ]
    
    get_remaining_master_sentences(reference_sentences, student_sentences, model) == [
        "AI applies Computer Vision",
        "Natural Language Processing supports Language Translation",
        "Automation improves Efficiency",
        "Machine Learning achieves Predictive Analytics",
        "Computer Vision applies Facial Recognition",
        "Computer Vision detects Objects",
        "Computer Vision analyzes Images",
        "Machine Learning utilizes Neural Networks",
        "Automation includes Robotics",
        "Automation integrates Internet of Things (IoT)",
        "Natural Language Processing enables Sentiment Analysis",
        "Machine Learning employs Algorithms",
        "Natural Language Processing enables Speech Recognition"]
    
