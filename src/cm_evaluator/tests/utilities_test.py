from cm_evaluator.utilities import assign_node_importance_weights, common_words, get_most_similar_word, get_remaining_master_sentences, words_are_similar
import networkx as nx

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
    
def test_words_are_similar(model):
    # base case identical words
    w1 = "dog"
    w2 = "dog"
    assert words_are_similar(w1, w2, model) == True
    
    w1 = "dog"
    w2 = "canine"
    assert words_are_similar(w1, w2, model) == True
    
    w1 = "car"
    w2 = "auto"
    assert words_are_similar(w1, w2, model) == True
    # case german
    
    w1 = "Auto"
    w2 = "Wagen"
    assert words_are_similar(w1, w2, model) == True
    
    w1 = 'zählt zu den'
    w2 = 'zählt zu den'
    assert words_are_similar(w1, w2, model, True) == True


def test_most_similar_word(model):
    word = 'AI'
    candidates = ['Computer Vision', 'Automation', 'AI']

    assert get_most_similar_word(word, candidates, model) == 'AI'

    # should work cross language
    word = 'AI'
    candidates = ['Computer Vision', 'Automation', 'KI']

    assert get_most_similar_word(word, candidates, model) == 'KI'

def test_common_words(model):
    words1 = ["cats", "mouse"]
    words2 = ["cats", "mouse"]

    assert common_words(words1, words2, model) == ["cats", "mouse"]

    words1 = ["cats", "mouse"]
    words2 = ["cats", "Maus"]

    assert common_words(words1, words2, model) == ["cats", "mouse"]

def test_assign_node_importance_weights():
    graph = nx.DiGraph()

    # test empty graph 
    assign_node_importance_weights(graph) == {}

    graph.add_edge("A", "B")
    assign_node_importance_weights(graph) == {"A": 0.3, "B": 0.7}

    graph.add_edge("B", "A")
    assign_node_importance_weights(graph) == {"A": 0.5, "B": 0.5}