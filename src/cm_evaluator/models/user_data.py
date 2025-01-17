from cm_evaluator.models.helper_models import MapRelation
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class Color(BaseModel):
    highlight: dict = Field(default_factory=dict)
    hover: dict = Field(default_factory=dict)
    border: str
    background: str

class Node(BaseModel):
    id: str
    label: str
    font: Optional[str]  = None
    shape: Optional[str] = None
    color: Optional[Color] = None
    x: Optional[int] = None
    y: Optional[int] = None

class ArrowOptions(BaseModel):
    enabled: bool

class Arrows(BaseModel):
    to: Optional[ArrowOptions]
    middle: Optional[ArrowOptions] = ArrowOptions(enabled=False)
    from_: Optional[ArrowOptions] = Field(default_factory=lambda: ArrowOptions(enabled=False), alias='from')

class EdgeColor(BaseModel):
    color: str
    highlight: str
    hover: str
    inherit: bool
    opacity: float

class Edge(BaseModel):
    id: str
    label: str
    arrows: Optional[Arrows] = None
    from_: str = Field(alias='from')
    to: str
    width: Optional[int] = 0
    dashes: Optional[bool] = False
    color: Optional[EdgeColor] = None

class MnmUserData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class StrictBaseModel(BaseModel):
    class Config:
        extra="forbid"

class MapOutput(StrictBaseModel):
    id: int
    mnm: int
    user: int
    mnm_user_data: MnmUserData
    timecreated: Optional[int] = 0
    timemodified: Optional[int] = 0
    lockedby_user: Optional[int] = 0

    def get_node_names(self) -> List[str]:
        return list(map(lambda x: x.label, self.mnm_user_data.nodes))

    def get_node_names_and_ids(self) -> Dict[str, str]:
        return {x.id: x.label for x in self.mnm_user_data.nodes}
    
    def convert_to_map_relation(self) -> List[MapRelation]:
        relations = []
        nodes = self.get_node_names_and_ids()
        for edge in self.mnm_user_data.edges:
            conceptA = nodes.get(edge.from_)
            conceptB = nodes.get(edge.to)
            relations.append(MapRelation(conceptA, edge.label, conceptB))
        return relations