from cm_evaluator.logging.base_logger import logger
import time
from typing import Callable


def time_and_log_decorator(func: Callable):
    def wrapper(*args, **kwargs):
        logger.debug(f'Calling function {func.__name__}')
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"The function {func.__name__} took {elapsed_time:.5f} seconds to execute. ")
        return result
    return wrapper
