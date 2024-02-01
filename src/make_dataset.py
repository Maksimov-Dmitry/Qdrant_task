import hydra
from omegaconf import DictConfig
from src.entities.make_dataset_params import read_make_dataset_params, MakeDatasetParams
from src.models.models import get_imagenet_model, generate_dataset
from src.dataset.dataset import CustomDataset
from dotenv import load_dotenv
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_dataset_pipeline(params: MakeDatasetParams):
    logger.info(f"Start making dataset with params: {params}")
    if 'gpt' not in params.dataset_model.model:
        model, transforms, device = get_imagenet_model(params.dataset_model)
        dataset = CustomDataset(params.input_directory, transforms)
    else:
        model = None
        device = None
        dataset = CustomDataset(params.input_directory, is_openai=True)
    train_dataset = generate_dataset(dataset, params, model, device)
    logger.info(f"Dataset was made with {len(train_dataset)} images")
    with open(params.output_file, 'w') as fout:
        for key, value in train_dataset.items():
            json_string = json.dumps({key: value})
            fout.write(json_string + '\n')


@hydra.main(version_base=None, config_path="../configs", config_name="dataset_config")
def make_dataset_command(cfg: DictConfig):
    params = read_make_dataset_params(cfg)
    load_dotenv()
    make_dataset_pipeline(params)


if __name__ == "__main__":
    make_dataset_command()
