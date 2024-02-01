from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class VectorDBParams:
    image_directory: str
    db: str
    collection_name: str
    model: str
    weights: str
    device: str
    batch_size: int


VectorDBParamsSchema = class_schema(VectorDBParams)


def read_vector_db_params(cfg: DictConfig) -> VectorDBParams:
    schema = VectorDBParamsSchema()
    return schema.load(cfg)
