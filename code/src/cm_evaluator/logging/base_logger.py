import logging
from pathlib import Path

logger = logging
file_path = Path(__file__).resolve().parent / "app.log"
logger.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    filename=file_path,
    encoding="utf-8",
    filemode="a",
    level=logging.DEBUG
)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)