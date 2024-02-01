import hydra
from omegaconf import DictConfig
from src.entities.evaluation_pipeline_params import read_evaluation_pipeline_params, EvaluatioPipelineParams
from src.dataset.dataset import CustomDataset
from src.models.models import get_clip_model, get_image_embeddings, get_text_embeddings
from dotenv import load_dotenv
from src.evaluation.evaluation import get_dataset, calculate_hit_rate_at_k, calculate_mrr
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluation_pipeline(params: EvaluatioPipelineParams):
    logger.info(f"Start evaluation pipeline with params: {params}")
    model, tokenizer, transforms, device = get_clip_model(params)
    images = CustomDataset(params.image_directory, transforms)
    image_embeddings = get_image_embeddings(params, model, device, images)
    dataset = get_dataset(params.dataset)
    text_embeddings = get_text_embeddings(model, tokenizer, device, dataset)
    similarity = np.matmul(text_embeddings, image_embeddings.T)
    relevant_indices = [images.image2idx[image] for image in dataset.keys()]
    mrr = calculate_mrr(similarity, relevant_indices)
    hit_rate_at_k = calculate_hit_rate_at_k(similarity, relevant_indices, params.top_k)
    result = {text: {'mrr': mrr[i], 'hr@k': hit_rate_at_k[i]} for i, text in enumerate(dataset.values())}
    with open(params.output_file, 'w') as fout:
        for key, value in result.items():
            json_string = json.dumps({key: value})
            fout.write(json_string + '\n')


@hydra.main(version_base=None, config_path="../configs", config_name="evaluation_config")
def evaluation_command(cfg: DictConfig):
    params = read_evaluation_pipeline_params(cfg)
    load_dotenv()
    evaluation_pipeline(params)


if __name__ == "__main__":
    evaluation_command()
