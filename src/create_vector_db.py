import hydra
from omegaconf import DictConfig
from src.entities.create_vector_db_params import read_vector_db_params, VectorDBParams
from src.dataset.dataset import CustomDataset
from src.models.models import get_clip_model, get_image_embeddings
import logging
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_vector_db(params: VectorDBParams):
    logger.info(f"Start creating vector db with params: {params}")
    model, _, transforms, device = get_clip_model(params)
    images = CustomDataset(params.image_directory, transforms)
    image_embeddings = get_image_embeddings(params, model, device, images)
    client = QdrantClient(path=params.db)
    client.recreate_collection(
        collection_name=params.collection_name,
        vectors_config=models.VectorParams(size=model.text.text_projection.out_features, distance=models.Distance.COSINE),
    )
    client.upload_points(
        collection_name=params.collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=image_embeddings[idx],
                payload={'path': str(image)},
            ) for idx, image in enumerate(images.images)
        ]
    )


@hydra.main(version_base=None, config_path="../configs", config_name="vector_db_config")
def create_vector_db_command(cfg: DictConfig):
    params = read_vector_db_params(cfg)
    create_vector_db(params)


if __name__ == "__main__":
    create_vector_db_command()
