from src.entities.imagenet_params import ImagenetParams
from src.entities.make_dataset_params import MakeDatasetParams
from src.entities.evaluation_pipeline_params import EvaluatioPipelineParams
from src.entities.create_vector_db_params import VectorDBParams
import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
import urllib
import os
from torch.utils.data import Subset
import random
import requests
import open_clip
import numpy as np
from typing import Union


def get_imagenet_model(params: ImagenetParams):
    device = torch.device(params.device)
    model = timm.create_model(params.model, pretrained=True)
    model = model.eval()
    model = model.to(device)
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return model, transforms, device


def _get_imagenet_classes(filename: str):
    if not os.path.exists(filename):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def _get_questions(image, max_tokens, temperature, top_p):
    prompt_template = """
        Analyze the following image and generate a short text that matches this image.
        If you believe the image is an advertisement, simply write the product being advertised here. Do not provide an explanation in any case!
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def generate_dataset(dataset, params: MakeDatasetParams, model=None, device=None):
    labeles = {}
    if params.subset is not None:
        subset = Subset(dataset, random.sample(range(len(dataset)), params.subset))
    else:
        subset = dataset
    if not dataset.is_openai:
        loader = DataLoader(subset, batch_size=params.dataset_model.batch_size, shuffle=False, num_workers=8)
        categories = _get_imagenet_classes('imagenet_classes.txt')
        with torch.no_grad():
            for tensor, idxs in tqdm(loader):
                tensor = tensor.to(device)
                out = model(tensor)
                probabilities = torch.nn.functional.softmax(out, dim=1)
                top_prob, top_catid = torch.topk(probabilities.to('cpu'), 1)
                mask = top_prob.squeeze() > params.dataset_model.threshold
                for catid, idx in zip(top_catid.squeeze()[mask], idxs[mask]):
                    labeles[str(dataset.images[idx.item()])] = categories[catid]
    else:
        for image, idx in tqdm(subset):
            try:
                response = _get_questions(image, params.dataset_model.max_tokens, params.dataset_model.temperature,
                                          params.dataset_model.top_p)
                labeles[str(dataset.images[idx])] = response['choices'][0]['message']['content']
            except Exception as e:
                print(e)

    return labeles


def get_clip_model(params: Union[EvaluatioPipelineParams, VectorDBParams]):
    device = torch.device(params.device)
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=params.model,
        pretrained=params.weights,
    )
    model = model.eval()
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(params.model)
    return model, tokenizer, transform, device


def get_image_embeddings(params: Union[EvaluatioPipelineParams, VectorDBParams], model, device, dataset):
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=8)
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for image, _ in tqdm(loader):
            image = image.to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.to('cpu').numpy())
    return np.vstack(embeddings)


def get_text_embeddings(model, tokenizer, device, dataset):
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for text in tqdm(dataset.values()):
            text = tokenizer(text).to(device)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            embeddings.append(text_features.to('cpu').numpy())
    return np.vstack(embeddings)
