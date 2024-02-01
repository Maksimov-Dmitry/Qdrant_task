from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class EvaluatioPipelineParams:
    dataset: str
    image_directory: str
    output_file: str
    top_k: int
    model: str
    weights: str
    device: str
    batch_size: int


EvaluatioPipelineParamsSchema = class_schema(EvaluatioPipelineParams)


def read_evaluation_pipeline_params(cfg: DictConfig) -> EvaluatioPipelineParams:
    schema = EvaluatioPipelineParamsSchema()
    return schema.load(cfg)
