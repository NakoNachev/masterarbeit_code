import tempfile
from typing import Literal, Optional

from cm_evaluator.graph_utils import create_graph_from_data
from fastapi import FastAPI, Response
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from cm_evaluator.models.user_data import MapOutput
import networkx as nx
from pyvis.network import Network
from cm_evaluator.evaluators import evaluatorA, evaluatorB, evaluatorC

app = FastAPI()
model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
# nlp_de = spacy.load("de_core_news_lg")

class MapCompareOptions(BaseModel):
    evaluator: Optional[Literal["A", "B", "C"]] = "C"
    language: Optional[Literal["de", "en"]] = "en"


class MapCompareBody(BaseModel):
    student_map: MapOutput
    reference_map: MapOutput
    options: Optional[MapCompareOptions] = None


@app.get("/")
async def get_root():
    return {"Hello": "World"}

@app.post("/")
async def create_item(map: MapOutput):
    return map

@app.post("/compare-maps")
async def compare_maps(body: MapCompareBody):
    module_mapper = {
        "A": evaluatorA,
        "B": evaluatorB,
        "C": evaluatorC
    }
    evaluator = body.options.evaluator if body.options else "C"
    language= body.options.language if body.options else "en"
    module = module_mapper.get(evaluator, None)
    if module:
        return module.evaluate(body.student_map, body.reference_map, model, language)

@app.post("/graph")
async def get_graph_as_html(map: MapOutput):

    relations = map.convert_to_map_relation()
    concepts = map.get_node_names()
    graph = create_graph_from_data(relations, concepts)
    
    net = Network(notebook=True, height="1300px", width="100%", directed=True)
    net.set_options("""
        {
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -20000,
            "centralGravity": 0.3,
            "springLength": 95,
            "springConstant": 0.04,
            "damping": 0.09,
            "avoidOverlap": 0.5
            },
            "solver": "barnesHut"
        },
        "edges": {
            "smooth": {
            "type": "curvedCW",
            "roundness": 0.5
            }
        }
        }
        """)
    net.from_nx(graph)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        net.show(temp_file.name)
        temp_file.seek(0)
        html_content = temp_file.read().decode("utf-8")
    
    return Response(content=html_content, media_type="text/html")