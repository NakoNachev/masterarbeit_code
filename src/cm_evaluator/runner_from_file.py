import json
import importlib
from importlib.resources import files

from cm_evaluator.models.user_data import MapOutput
from sentence_transformers import SentenceTransformer

from cm_evaluator.logging.base_logger import logger

def main():

    type='C'
    module_mapper = {
        "A": "evaluators.evaluatorA",
        "B": "evaluators.evaluatorB",
        "C": "evaluators.evaluatorC"
    }

    cm_student = None
    cm_master = None
    model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
    logger.debug(f'Starting evaluation for type {type} and user_file')

    target_dir = "cm_evaluator.data.datasets.sample_dataset"
    reference_file = "reference_solution_en.json"
    student_file = "student_solution_minimal_en.json"
    language = "en"

    with open(files(target_dir).joinpath(reference_file), 'r', encoding='utf-8') as fd:
        t = json.load(fd)
        cm_master = MapOutput(**t)

    with open(files(target_dir).joinpath(student_file), 'r', encoding='utf-8') as fd:
        t = json.load(fd)
        cm_student = MapOutput(**t)

    # run the evaluator function for the chosen evaluator
    module = importlib.import_module(module_mapper.get(type, None), )
    module.evaluate(cm_student, cm_master, model, language, True)
    logger.debug(f'Finished evaluating user_file')

if __name__ == '__main__':
    main()