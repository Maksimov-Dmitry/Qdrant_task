from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import Union
from src.entities.imagenet_params import ImagenetParams
from src.entities.openai_params import OpenAIParams
from typing import Optional


@dataclass()
class MakeDatasetParams:
    dataset_model: Union[ImagenetParams, OpenAIParams]
    input_directory: str
    output_file: str
    subset: Optional[int]


MakeDatasetParamsSchema = class_schema(MakeDatasetParams)


def read_make_dataset_params(cfg: DictConfig) -> MakeDatasetParams:
    schema = MakeDatasetParamsSchema()
    return schema.load(cfg)
